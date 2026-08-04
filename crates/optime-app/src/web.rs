//! Browser-only helpers: file download, demo fetching, and the shareable track URL.

use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;
use web_sys::window;

use crate::persisted::TrackRef;

pub async fn fetch_bytes(url: &str) -> Option<Vec<u8>> {
    let window = web_sys::window()?;
    let opts = web_sys::RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(web_sys::RequestMode::SameOrigin);

    let request = web_sys::Request::new_with_str_and_init(url, &opts).ok()?;
    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .ok()?;
    let resp: web_sys::Response = resp_value.dyn_into().ok()?;
    if !resp.ok() {
        return None;
    }
    let buf = JsFuture::from(resp.array_buffer().ok()?).await.ok()?;
    let array = js_sys::Uint8Array::new(&buf);
    Some(array.to_vec())
}

pub fn download(filename: &str, bytes: &[u8]) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };

    let array = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::new();
    parts.push(&array.buffer());

    let opts = web_sys::BlobPropertyBag::new();
    opts.set_type("audio/wav");

    let Ok(blob) = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &opts) else {
        return;
    };
    let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) else {
        return;
    };

    if let Ok(anchor) = document.create_element("a")
        && let Ok(anchor) = anchor.dyn_into::<web_sys::HtmlAnchorElement>()
    {
        anchor.set_href(&url);
        anchor.set_download(filename);
        anchor.click();
    }
    let _ = web_sys::Url::revoke_object_url(&url);
}

pub fn get_track_ref_from_query_string() -> Option<TrackRef> {
    let window = window().ok_or("No global window exists").ok()?;

    let location = window.location();

    let search = location.search().ok()?;

    let url_params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;

    Some(TrackRef {
        source: url_params.get("source")?.clone(),
        song_id: url_params.get("sseq_id")?.parse::<u32>().ok()?,
        label: url_params.get("label")?.clone(),
    })
}

pub fn update_query_string(track_ref: TrackRef) -> Result<(), JsValue> {
    let window = window().expect("no global `window` exists");
    let history = window.history()?;

    let new_query = format!(
        "?source={}&sseq_id={}&label={}",
        track_ref.source, track_ref.song_id, track_ref.label
    );

    let location = window.location();
    let pathname = location.pathname()?;
    let new_url = format!("{}{}", pathname, new_query);

    history.replace_state_with_url(&JsValue::NULL, "", Some(&new_url))?;

    Ok(())
}
