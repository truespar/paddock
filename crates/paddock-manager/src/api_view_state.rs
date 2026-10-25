use super::*;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new().route(
        "/api/conversations/{id}/view-state/{scope}",
        get(load).put(save),
    )
}
async fn load(
    State(s): State<Arc<AppState>>,
    Path((id, scope)): Path<(String, String)>,
) -> Response {
    match s.db.get_conversation_view_state(&id, &scope) {
        Ok(value) => Json(value).into_response(),
        Err(crate::store::StoreError::Bad(e)) => {
            errx(StatusCode::BAD_REQUEST, "invalid_request", e)
        }
        Err(e) => err500(e),
    }
}
async fn save(
    State(s): State<Arc<AppState>>,
    Path((id, scope)): Path<(String, String)>,
    Json(doc): Json<Value>,
) -> Response {
    match s.db.put_conversation_view_state(&id, &scope, &doc) {
        Ok(saved) => Json(json!({"saved": saved})).into_response(),
        Err(crate::store::StoreError::Bad(e)) => {
            errx(StatusCode::BAD_REQUEST, "invalid_request", e)
        }
        Err(e) => err500(e),
    }
}
