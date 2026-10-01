use super::*;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/table-history", get(list))
        .route(
            "/api/table-history/{id}",
            get(load).put(save).delete(remove),
        )
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
}

#[derive(serde::Deserialize)]
struct Revision {
    revision: String,
}

async fn list(State(s): State<Arc<AppState>>) -> Response {
    match tokio::task::spawn_blocking(move || s.db.list_table_history()).await {
        Ok(Ok(rows)) => Json(rows).into_response(),
        Ok(Err(e)) => prompt_error(e),
        Err(e) => err500(e),
    }
}
async fn load(State(s): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match tokio::task::spawn_blocking(move || s.db.get_table_session(&id)).await {
        Ok(Ok(Some(doc))) => Json(doc).into_response(),
        Ok(Ok(None)) => errx(StatusCode::NOT_FOUND, "not_found", "Table not found"),
        Ok(Err(e)) => prompt_error(e),
        Err(e) => err500(e),
    }
}
async fn save(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<Revision>,
    text: String,
) -> Response {
    match tokio::task::spawn_blocking(move || s.db.put_table_session(&id, &text, &q.revision)).await
    {
        Ok(Ok(row)) => Json(row).into_response(),
        Ok(Err(e)) => prompt_error(e),
        Err(e) => err500(e),
    }
}
async fn remove(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<Revision>,
) -> Response {
    match tokio::task::spawn_blocking(move || s.db.delete_table_session(&id, &q.revision)).await {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        Ok(Err(e)) => prompt_error(e),
        Err(e) => err500(e),
    }
}
