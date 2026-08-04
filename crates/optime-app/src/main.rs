fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let native_options = eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default()
                .with_inner_size([1100.0, 720.0])
                .with_title("Optime Player"),
            ..Default::default()
        };
        eframe::run_native(
            "Optime Player",
            native_options,
            Box::new(|cc| Ok(Box::new(optime_app::OptimeApp::new(cc)))),
        )
        .expect("failed to start eframe");
    }

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        console_error_panic_hook::set_once();
        let web_options = eframe::WebOptions::default();
        wasm_bindgen_futures::spawn_local(async {
            let canvas = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("the_canvas_id"))
                .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                .expect("missing #the_canvas_id");
            eframe::WebRunner::new()
                .start(
                    canvas,
                    web_options,
                    Box::new(|cc| Ok(Box::new(optime_app::OptimeApp::new(cc)))),
                )
                .await
                .expect("failed to start eframe on web");
        });
    }
}
