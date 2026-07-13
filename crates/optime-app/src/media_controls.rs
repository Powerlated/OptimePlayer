//! OS media-transport integration: the system "now playing" controls and hardware/Bluetooth media
//! keys (e.g. double-tapping AirPods to skip a track, or the keyboard's play/pause key).
//!
//! Native builds use [`souvlaki`] — System Media Transport Controls on Windows, MPRIS on Linux,
//! the Now Playing center on macOS. The web build uses the browser's [Media Session API]
//! (`navigator.mediaSession`), which drives the phone lock-screen / notification transport and
//! Bluetooth/headset media keys on mobile. The app polls [`MediaControls::poll`] each frame for any
//! [`MediaAction`]s the user triggered and pushes the current track/playback state back with
//! [`MediaControls::set_now_playing`].
//!
//! [Media Session API]: https://developer.mozilla.org/en-US/docs/Web/API/Media_Session_API

/// A transport command coming *from* the OS (a media key / Now Playing button press).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// The web build's stub never constructs these (it has no OS transport), but native does.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub enum MediaAction {
    /// Toggle play/pause (the usual single media-key / AirPods single-tap).
    PlayPause,
    Play,
    Pause,
    /// Skip to the next track (AirPods double-tap, or the "next" media key).
    Next,
    /// Skip to the previous track (AirPods triple-tap, or the "previous" media key).
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

    /// Owns the OS media controls and the channel its event callback feeds.
    pub struct MediaControls {
        controls: Smtc,
        events: Receiver<MediaAction>,
        /// The last `(title, playing)` pushed, to avoid re-sending unchanged metadata every frame.
        last: Option<(String, bool)>,
    }

    impl MediaControls {
        /// Builds the controls bound to the app window (`frame` supplies the native window handle,
        /// required for the Windows transport controls). Returns `None` if the handle or the OS
        /// controls are unavailable. `ctx` is unused natively (the OS callback already wakes the
        /// event loop); the web build uses it to repaint when a media key fires.
        pub fn new(_ctx: &egui::Context, frame: &eframe::Frame) -> Option<Self> {
            // Windows' SMTC needs the host HWND; other platforms ignore it.
            let hwnd = match frame.window_handle().ok()?.as_raw() {
                RawWindowHandle::Win32(h) => Some(h.hwnd.get() as *mut std::ffi::c_void),
                _ => None,
            };
            #[cfg(target_os = "windows")]
            hwnd?; // SMTC cannot be created without the window handle.

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

        /// Drains any transport commands the user triggered since the last call.
        pub fn poll(&mut self) -> Vec<MediaAction> {
            self.events.try_iter().collect()
        }

        /// Updates the system "now playing" title and play/pause state. Cheap to call every frame:
        /// it only talks to the OS when the title or playing state actually changed.
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

    /// Transport commands the Media Session callbacks push for the next `poll`. wasm is
    /// single-threaded, so an `Rc<RefCell<…>>` shared with the JS closures is enough.
    type Queue = Rc<RefCell<Vec<MediaAction>>>;

    /// Drives the browser's `navigator.mediaSession`: the phone lock-screen transport plus
    /// hardware/Bluetooth media keys.
    pub struct MediaControls {
        session: MediaSession,
        queue: Queue,
        /// The action-handler closures, kept alive for as long as the session references them.
        _handlers: Vec<Closure<dyn FnMut()>>,
        /// The last `(title, playing)` pushed, to avoid re-sending unchanged metadata every frame.
        last: Option<(String, bool)>,
    }

    impl MediaControls {
        pub fn new(ctx: &egui::Context, _frame: &eframe::Frame) -> Option<Self> {
            let session = web_sys::window()?.navigator().media_session();
            let queue: Queue = Rc::new(RefCell::new(Vec::new()));
            let mut handlers: Vec<Closure<dyn FnMut()>> = Vec::new();
            {
                // Register one handler per supported action. Each pushes its command and wakes the
                // egui repaint clock, so a key press is applied on the very next frame even while
                // the UI is idling at a low refresh rate.
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
                // The Media Session API has no single "toggle"; play and pause are separate.
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

        /// Drains any transport commands the user triggered since the last call.
        pub fn poll(&mut self) -> Vec<MediaAction> {
            std::mem::take(&mut *self.queue.borrow_mut())
        }

        /// Updates the system "now playing" title and play/pause state. Cheap to call every frame:
        /// it only talks to the browser when the title or playing state actually changed.
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
