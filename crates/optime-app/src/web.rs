//! Web-only helpers (browser file download and demo fetching).

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

/// Fetches `url` (relative to the page) and returns its bytes, or `None` on any error.
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

/// Triggers a browser download of `bytes` as `filename` via an object-URL anchor.
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

    if let Ok(anchor) = document.create_element("a") {
        if let Ok(anchor) = anchor.dyn_into::<web_sys::HtmlAnchorElement>() {
            anchor.set_href(&url);
            anchor.set_download(filename);
            anchor.click();
        }
    }
    let _ = web_sys::Url::revoke_object_url(&url);
}
