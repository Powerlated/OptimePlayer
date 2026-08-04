//! Media-key integration: the system transport natively, the Media Session API on the web.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub enum MediaAction {
    PlayPause,
    Play,
    Pause,
    Next,
    Prev,
    Stop,
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::MediaControls;

#[cfg(target_arch = "wasm32")]
pub use web::MediaControls;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::mpsc::{Receiver, channel};

    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use souvlaki::{
        MediaControlEvent, MediaControls as Smtc, MediaMetadata, MediaPlayback, PlatformConfig,
    };

    use super::MediaAction;

    pub struct MediaControls {
        controls: Smtc,
        events: Receiver<MediaAction>,
        last: Option<(String, bool)>,
    }

    impl MediaControls {
        pub fn new(_ctx: &egui::Context, frame: &eframe::Frame) -> Option<Self> {
            let hwnd = match frame.window_handle().ok()?.as_raw() {
                RawWindowHandle::Win32(h) => Some(h.hwnd.get() as *mut std::ffi::c_void),
                _ => None,
            };
            #[cfg(target_os = "windows")]
            hwnd?;

            let config = PlatformConfig {
                dbus_name: "optime_player",
                display_name: "Optime Player",
                hwnd,
            };
            let mut controls = Smtc::new(config).ok()?;

            let (tx, rx) = channel();
            controls
                .attach(move |event: MediaControlEvent| {
                    let action = match event {
                        MediaControlEvent::Toggle => Some(MediaAction::PlayPause),
                        MediaControlEvent::Play => Some(MediaAction::Play),
                        MediaControlEvent::Pause => Some(MediaAction::Pause),
                        MediaControlEvent::Next => Some(MediaAction::Next),
                        MediaControlEvent::Previous => Some(MediaAction::Prev),
                        MediaControlEvent::Stop | MediaControlEvent::Quit => {
                            Some(MediaAction::Stop)
                        }
                        _ => None,
                    };
                    if let Some(a) = action {
                        let _ = tx.send(a);
                    }
                })
                .ok()?;

            Some(Self {
                controls,
                events: rx,
                last: None,
            })
        }

        pub fn poll(&mut self) -> Vec<MediaAction> {
            self.events.try_iter().collect()
        }

        pub fn set_now_playing(&mut self, title: &str, artist: &str, playing: bool) {
            if self
                .last
                .as_ref()
                .is_some_and(|(t, p)| t == title && *p == playing)
            {
                return;
            }
            let _ = self.controls.set_metadata(MediaMetadata {
                title: Some(title),
                artist: Some(artist),
                ..MediaMetadata::default()
            });
            let _ = self.controls.set_playback(if playing {
                MediaPlayback::Playing { progress: None }
            } else {
                MediaPlayback::Paused { progress: None }
            });
            self.last = Some((title.to_owned(), playing));
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use std::cell::RefCell;
    use std::rc::Rc;

    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;
    use web_sys::{MediaMetadata, MediaSession, MediaSessionAction, MediaSessionPlaybackState};

    use super::MediaAction;

    type Queue = Rc<RefCell<Vec<MediaAction>>>;

    pub struct MediaControls {
        session: MediaSession,
        queue: Queue,
        _handlers: Vec<Closure<dyn FnMut()>>,
        last: Option<(String, bool)>,
    }

    impl MediaControls {
        pub fn new(ctx: &egui::Context, _frame: &eframe::Frame) -> Option<Self> {
            let session = web_sys::window()?.navigator().media_session();
            let queue: Queue = Rc::new(RefCell::new(Vec::new()));
            let mut handlers: Vec<Closure<dyn FnMut()>> = Vec::new();
            {
                let mut bind = |action: MediaSessionAction, ev: MediaAction| {
                    let queue = queue.clone();
                    let ctx = ctx.clone();
                    let cb = Closure::wrap(Box::new(move || {
                        queue.borrow_mut().push(ev);
                        ctx.request_repaint();
                    }) as Box<dyn FnMut()>);
                    session.set_action_handler(action, Some(cb.as_ref().unchecked_ref()));
                    handlers.push(cb);
                };
                bind(MediaSessionAction::Play, MediaAction::Play);
                bind(MediaSessionAction::Pause, MediaAction::Pause);
                bind(MediaSessionAction::Previoustrack, MediaAction::Prev);
                bind(MediaSessionAction::Nexttrack, MediaAction::Next);
                bind(MediaSessionAction::Stop, MediaAction::Stop);
            }

            Some(Self {
                session,
                queue,
                _handlers: handlers,
                last: None,
            })
        }

        pub fn poll(&mut self) -> Vec<MediaAction> {
            std::mem::take(&mut *self.queue.borrow_mut())
        }

        pub fn set_now_playing(&mut self, title: &str, artist: &str, playing: bool) {
            if self
                .last
                .as_ref()
                .is_some_and(|(t, p)| t == title && *p == playing)
            {
                return;
            }
            if let Ok(meta) = MediaMetadata::new() {
                meta.set_title(title);
                meta.set_artist(artist);
                self.session.set_metadata(Some(&meta));
            }
            self.session.set_playback_state(if playing {
                MediaSessionPlaybackState::Playing
            } else {
                MediaSessionPlaybackState::Paused
            });
            self.last = Some((title.to_owned(), playing));
        }
    }
}
