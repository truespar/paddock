use super::*;
fn fixture() -> Value {
    let source = "size,city,label\n1,a,yes\n2,b,no\n3,c,\n";
    let key = hash(source);
    let input = json!({"dataset":key,"fileName":"Example.csv","spec":{"target":2,"use":[true,true,true],"types":["numerical","categorical","categorical"]},"model":"kumo-tabular-large-classification","port":11544,"estimators":8,"seed":0});
    json!({"version":1,"id":"table-a","title":"Example.csv","model":"kumo","createdAt":1,"updatedAt":2,
        "draft":input,"datasets":{key:source},"runs":[]})
}
fn run(doc: &Value, id: &str) -> Value {
    json!({"id":id,"at":2,"input":doc["draft"],"task":"classification","ms":12.5,
        "response":{"task":"classification","classes":["yes","no"],"predictions":[{"class":0,"label":"yes","probabilities":[0.8,0.2]}]}})
}
#[test]
fn roundtrip_reopen_summary_and_cross_client_conflicts() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("tables.db");
    let native = Store::open(&path).unwrap();
    let web = Store::open(&path).unwrap();
    let mut doc = fixture();
    let first = native
        .put_table_session("table-a", &doc.to_string(), "")
        .unwrap();
    doc["runs"] = json!([run(&doc, "run-a")]);
    let next = web
        .put_table_session(
            "table-a",
            &doc.to_string(),
            first["revision"].as_str().unwrap(),
        )
        .unwrap();
    assert!(matches!(
        native.put_table_session(
            "table-a",
            &fixture().to_string(),
            first["revision"].as_str().unwrap()
        ),
        Err(StoreError::Conflict(_))
    ));
    assert!(matches!(
        native.delete_table_session("table-a", first["revision"].as_str().unwrap()),
        Err(StoreError::Conflict(_))
    ));
    // Lost acknowledgements are safe to retry, with the old revision.
    let retry = native
        .put_table_session(
            "table-a",
            &doc.to_string(),
            first["revision"].as_str().unwrap(),
        )
        .unwrap();
    assert_eq!(retry, next);
    let list = native.list_table_history().unwrap();
    assert_eq!(list[0]["runs"], 1);
    assert!(list[0].get("datasets").is_none());
    assert!(
        native
            .put_table_session(
                "table-a",
                &list[0].to_string(),
                next["revision"].as_str().unwrap()
            )
            .is_err()
    );
    drop(native);
    drop(web);
    let reopened = Store::open(&path).unwrap();
    let saved = reopened.get_table_session("table-a").unwrap().unwrap();
    assert_eq!(saved["doc"], doc.to_string());
    reopened
        .delete_table_session("table-a", next["revision"].as_str().unwrap())
        .unwrap();
    assert!(reopened.list_table_history().unwrap().is_empty());
    assert!(
        reopened
            .put_table_session(
                "table-a",
                &doc.to_string(),
                next["revision"].as_str().unwrap()
            )
            .is_err()
    );
}
#[test]
fn datasets_are_content_addressed_and_runs_cannot_be_rewritten() {
    let db = Store::open(&PathBuf::from(":memory:")).unwrap();
    let mut doc = fixture();
    doc["runs"] = json!([run(&doc, "run-a")]);
    let row = db
        .put_table_session("table-a", &doc.to_string(), "")
        .unwrap();
    let revision = row["revision"].as_str().unwrap();
    let mut bad = doc.clone();
    bad["runs"][0]["ms"] = json!(99);
    assert!(
        db.put_table_session("table-a", &bad.to_string(), revision)
            .is_err()
    );
    bad = doc.clone();
    bad["runs"] = json!([]);
    assert!(
        db.put_table_session("table-a", &bad.to_string(), revision)
            .is_err()
    );
    bad = doc.clone();
    bad["datasets"][doc["draft"]["dataset"].as_str().unwrap()] = json!("different");
    assert!(
        db.put_table_session("table-a", &bad.to_string(), revision)
            .is_err()
    );
    doc["title"] = json!("Renamed");
    let row = db
        .put_table_session("table-a", &doc.to_string(), revision)
        .unwrap();
    assert_eq!(row["title"], "Renamed");
    assert_eq!(row["runs"], 1);
}
#[test]
fn rejects_invalid_contracts_without_modifying_saved_data() {
    let db = Store::open(&PathBuf::from(":memory:")).unwrap();
    for (field, value) in [
        ("version", json!(2)),
        ("runs", json!(0)),
        ("datasets", json!({})),
        ("draft", json!({})),
        ("id", json!("wrong")),
        ("title", json!("")),
    ] {
        let mut doc = fixture();
        doc[field] = value;
        assert!(
            db.put_table_session("table-a", &doc.to_string(), "")
                .is_err(),
            "{field}"
        );
    }
    let mut doc = fixture();
    doc["runs"] = json!([run(&doc, "same"), run(&doc, "same")]);
    assert!(
        db.put_table_session("table-a", &doc.to_string(), "")
            .is_err()
    );
    assert!(db.list_table_history().unwrap().is_empty());
}
