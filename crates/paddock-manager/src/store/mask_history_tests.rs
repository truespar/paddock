use super::*;

const PNG: &str = "data:image/png;base64,iVBORw0KGgo=";

fn fixture() -> Value {
    let key = hash(PNG);
    json!({"version":1,"id":"masks-a","title":"street.png","model":"sam3","createdAt":1,"updatedAt":2,
        "pictures":{key.clone():PNG},
        "picture":{"ref":key,"name":"street.png","width":640,"height":480},
        "layers":[{"kind":"concept","text":"person","boxes":[],"points":[],"objectBox":null,
            "threshold":0.5,"visible":true,"choice":0,"hidden":[],"result":null}],
        "active":0,"snapshots":[]})
}

#[test]
fn roundtrip_counts_found_prompts_and_refuses_stale_writes() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("masks.db");
    let a = Store::open(&path).unwrap();
    let b = Store::open(&path).unwrap();
    let mut doc = fixture();
    let first = a.put_mask_session("masks-a", &doc.to_string(), "").unwrap();
    assert_eq!(first["runs"], 0);
    doc["layers"][0]["result"] = json!({"object":"masks","instances":[]});
    doc["updatedAt"] = json!(3);
    let text = doc.to_string();
    let next = b
        .put_mask_session("masks-a", &text, first["revision"].as_str().unwrap())
        .unwrap();
    assert_eq!(next["runs"], 1);
    // the text comes back byte for byte
    let got = a.get_mask_session("masks-a").unwrap().unwrap();
    assert_eq!(got["doc"].as_str(), Some(text.as_str()));
    // a write made on the first revision is a conflict now...
    assert!(matches!(
        a.put_mask_session(
            "masks-a",
            &fixture().to_string(),
            first["revision"].as_str().unwrap()
        ),
        Err(StoreError::Conflict(_))
    ));
    // ...unless it is the same text again (a lost reply)
    let retry = a
        .put_mask_session("masks-a", &text, first["revision"].as_str().unwrap())
        .unwrap();
    assert_eq!(retry, next);
    // the list carries no document
    let list = a.list_mask_history().unwrap();
    assert_eq!(list.len(), 1);
    assert!(list[0].get("pictures").is_none());
    assert!(matches!(
        a.delete_mask_session("masks-a", first["revision"].as_str().unwrap()),
        Err(StoreError::Conflict(_))
    ));
    a.delete_mask_session("masks-a", next["revision"].as_str().unwrap())
        .unwrap();
    assert!(a.list_mask_history().unwrap().is_empty());
}

#[test]
fn refuses_documents_the_list_or_a_reopen_cannot_read() {
    let root = tempfile::tempdir().unwrap();
    let db = Store::open(&root.path().join("masks.db")).unwrap();
    let put = |d: &Value| db.put_mask_session("masks-a", &d.to_string(), "");
    // a picture keyed by anything but its own hash
    let mut d = fixture();
    let key = hash(PNG);
    d["pictures"] = json!({"0000": PNG});
    assert!(put(&d).is_err());
    // a picture that is not a picture
    d = fixture();
    d["pictures"] = json!({hash("data:text/html,x"): "data:text/html,x"});
    d["picture"]["ref"] = json!(hash("data:text/html,x"));
    assert!(put(&d).is_err());
    // a snapshot whose frame is not in the record
    d = fixture();
    d["snapshots"] = json!([{"id":"s1","ref":"missing","width":640,"height":360,"at":5,"concepts":[],"objects":[]}]);
    assert!(put(&d).is_err());
    // a list row sent back in place of the record
    d = json!({"id":"masks-a","title":"street.png","model":"sam3","runs":1,"createdAt":1,"updatedAt":2});
    assert!(put(&d).is_err());
    // the record itself, with its snapshot, is fine
    d = fixture();
    d["snapshots"] = json!([{"id":"s1","ref":key,"width":640,"height":480,"at":5,"concepts":[{"text":"person","slot":0}],"objects":[]}]);
    d["picture"] = Value::Null;
    assert!(put(&d).is_ok());
}
