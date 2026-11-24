#[cfg(unix)]
mod imp {
    use std::ffi::CString;
    use std::io::{self, Write};
    use std::os::raw::{c_char, c_int, c_short, c_uchar, c_uint, c_ulong, c_void};
    use std::path::Path;
    use std::time::{Duration, Instant};

    const STDIN_FD: c_int = 0;

    const O_RDWR: c_int = 0o2;
    const O_NOCTTY: c_int = 0o400;
    const O_NONBLOCK: c_int = 0o4000;
    const F_GETFL: c_int = 3;
    const F_SETFL: c_int = 4;

    const TCSANOW: c_int = 0;
    const TIOCMBIS: c_ulong = 0x5416;
    const TIOCM_DTR: c_int = 0x2;
    const TIOCM_RTS: c_int = 0x4;

    const NCCS: usize = 32;
    const VTIME: usize = 5;
    const VMIN: usize = 6;

    const IGNBRK: c_uint = 0x1;
    const BRKINT: c_uint = 0x2;
    const PARMRK: c_uint = 0x8;
    const ISTRIP: c_uint = 0x20;
    const INLCR: c_uint = 0x40;
    const IGNCR: c_uint = 0x80;
    const ICRNL: c_uint = 0x100;
    const IXON: c_uint = 0x400;
    const IXOFF: c_uint = 0x1000;

    const OPOST: c_uint = 0x1;

    const CSIZE: c_uint = 0x30;
    const CS8: c_uint = 0x30;
    const CSTOPB: c_uint = 0x40;
    const CREAD: c_uint = 0x80;
    const PARENB: c_uint = 0x100;
    const HUPCL: c_uint = 0x400;
    const CLOCAL: c_uint = 0x800;

    const ISIG: c_uint = 0x1;
    const ICANON: c_uint = 0x2;
    const ECHO: c_uint = 0x8;
    const ECHOE: c_uint = 0x10;
    const ECHOK: c_uint = 0x20;
    const ECHONL: c_uint = 0x40;
    const IEXTEN: c_uint = 0x8000;

    const POLLIN: c_short = 0x1;
    const POLLERR: c_short = 0x8;
    const POLLHUP: c_short = 0x10;

    const SPEED_115200_MODERN: c_uint = 115_200;
    const SPEED_115200_LEGACY: c_uint = 0o010002;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Termios {
        c_iflag: c_uint,
        c_oflag: c_uint,
        c_cflag: c_uint,
        c_lflag: c_uint,
        c_line: c_uchar,
        c_cc: [c_uchar; NCCS],
        c_ispeed: c_uint,
        c_ospeed: c_uint,
    }

    impl Default for Termios {
        fn default() -> Self {
            Self {
                c_iflag: 0,
                c_oflag: 0,
                c_cflag: 0,
                c_lflag: 0,
                c_line: 0,
                c_cc: [0; NCCS],
                c_ispeed: 0,
                c_ospeed: 0,
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PollFd {
        fd: c_int,
        events: c_short,
        revents: c_short,
    }

    extern "C" {
        fn open(pathname: *const c_char, flags: c_int, ...) -> c_int;
        fn close(fd: c_int) -> c_int;
        fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
        fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
        fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int;
        fn isatty(fd: c_int) -> c_int;
        fn tcgetattr(fd: c_int, termios_p: *mut Termios) -> c_int;
        fn tcsetattr(fd: c_int, optional_actions: c_int, termios_p: *const Termios) -> c_int;
        fn cfsetispeed(termios_p: *mut Termios, speed: c_uint) -> c_int;
        fn cfsetospeed(termios_p: *mut Termios, speed: c_uint) -> c_int;
        fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
        fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    }

    pub fn stdin_is_tty() -> bool {
        unsafe { isatty(STDIN_FD) == 1 }
    }

    /// Guard that puts local stdin into character-at-a-time mode when attached to a TTY and
    /// restores the original `termios` settings on drop.
    pub struct InteractiveTtyGuard {
        orig: Option<Termios>,
    }

    impl InteractiveTtyGuard {
        pub fn enter(keep_isig: bool) -> Self {
            if !stdin_is_tty() {
                return Self { orig: None };
            }
            let mut orig = Termios::default();
            if unsafe { tcgetattr(STDIN_FD, &mut orig) } != 0 {
                return Self { orig: None };
            }
            let mut raw = orig;
            raw.c_lflag &= !(ICANON | ECHO | ECHOE | ECHOK | ECHONL | IEXTEN);
            if keep_isig {
                raw.c_lflag |= ISIG;
            } else {
                raw.c_lflag &= !ISIG;
                raw.c_iflag &= !(BRKINT | IXON | IXOFF | ICRNL);
            }
            raw.c_cc[VMIN] = 1;
            raw.c_cc[VTIME] = 0;
            if unsafe { tcsetattr(STDIN_FD, TCSANOW, &raw) } != 0 {
                return Self { orig: None };
            }
            Self { orig: Some(orig) }
        }
    }

    impl Drop for InteractiveTtyGuard {
        fn drop(&mut self) {
            if let Some(orig) = &self.orig {
                unsafe {
                    let _ = tcsetattr(STDIN_FD, TCSANOW, orig);
                }
            }
        }
    }

    pub struct SerialPort {
        fd: c_int,
    }

    impl SerialPort {
        pub fn poll_read(&self, buf: &mut [u8], timeout_ms: i32) -> Result<Option<usize>, String> {
            let mut pfd = PollFd {
                fd: self.fd,
                events: POLLIN,
                revents: 0,
            };
            let rc = loop {
                let r = unsafe { poll(&mut pfd, 1, timeout_ms) };
                if r < 0 {
                    let err = io::Error::last_os_error();
                    if err.kind() == io::ErrorKind::Interrupted {
                        continue;
                    }
                    return Err(format!("serial poll failed: {err}"));
                }
                break r;
            };
            if rc == 0 {
                return Ok(None);
            }
            if (pfd.revents & (POLLERR | POLLHUP)) != 0 && (pfd.revents & POLLIN) == 0 {
                return Err("serial device disconnected".to_string());
            }
            let n = loop {
                let r = unsafe { read(self.fd, buf.as_mut_ptr().cast::<c_void>(), buf.len()) };
                if r < 0 {
                    let err = io::Error::last_os_error();
                    if err.kind() == io::ErrorKind::Interrupted {
                        continue;
                    }
                    if err.kind() == io::ErrorKind::WouldBlock {
                        return Ok(None);
                    }
                    return Err(format!("serial read failed: {err}"));
                }
                break r;
            };
            if n == 0 {
                return Err("serial device closed".to_string());
            }
            Ok(Some(n as usize))
        }

        pub fn write_all_bytes(&self, mut data: &[u8]) -> Result<(), String> {
            while !data.is_empty() {
                let n = unsafe { write(self.fd, data.as_ptr().cast::<c_void>(), data.len()) };
                if n < 0 {
                    let err = io::Error::last_os_error();
                    if err.kind() == io::ErrorKind::Interrupted {
                        continue;
                    }
                    return Err(format!("serial write failed: {err}"));
                }
                data = &data[n as usize..];
            }
            Ok(())
        }
    }

    impl Drop for SerialPort {
        fn drop(&mut self) {
            unsafe {
                let _ = close(self.fd);
            }
        }
    }

    fn set_baud_115200(t: &mut Termios) -> Result<(), String> {
        // glibc >= 2.42 uses numeric baud constants (`115200U`), while older glibc uses octal
        // `__B115200` (`0o010002`); try numeric first so newer glibc does not see `4098`.
        let mut set_pair = |speed: c_uint| -> bool {
            let mut tmp = *t;
            let rc =
                unsafe { cfsetispeed(&mut tmp, speed) == 0 && cfsetospeed(&mut tmp, speed) == 0 };
            if rc {
                *t = tmp;
            }
            rc
        };
        if set_pair(SPEED_115200_MODERN) || set_pair(SPEED_115200_LEGACY) {
            Ok(())
        } else {
            Err("failed to set serial baud rate to 115200".to_string())
        }
    }

    pub fn open_serial_115200(port_path: &str) -> Result<SerialPort, String> {
        let c_path = CString::new(port_path)
            .map_err(|_| format!("invalid serial port path `{port_path}`"))?;
        let fd = unsafe { open(c_path.as_ptr(), O_RDWR | O_NOCTTY | O_NONBLOCK) };
        if fd < 0 {
            let err = io::Error::last_os_error();
            return Err(format!("failed to open serial port `{port_path}`: {err}"));
        }

        let port = SerialPort { fd };

        let flags = unsafe { fcntl(fd, F_GETFL) };
        if flags >= 0 {
            let _ = unsafe { fcntl(fd, F_SETFL, flags & !O_NONBLOCK) };
        }

        let mut t = Termios::default();
        if unsafe { tcgetattr(fd, &mut t) } != 0 {
            let err = io::Error::last_os_error();
            return Err(format!(
                "failed to query serial attributes on `{port_path}`: {err}"
            ));
        }

        t.c_iflag &= !(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON | IXOFF);
        t.c_oflag &= !OPOST;
        t.c_lflag &= !(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
        t.c_cflag &= !(CSIZE | PARENB | CSTOPB | HUPCL);
        t.c_cflag |= CS8 | CREAD | CLOCAL;
        t.c_cc[VMIN] = 0;
        t.c_cc[VTIME] = 0;
        set_baud_115200(&mut t)?;

        if unsafe { tcsetattr(fd, TCSANOW, &t) } != 0 {
            let err = io::Error::last_os_error();
            return Err(format!(
                "failed to configure serial port `{port_path}` (115200 8N1): {err}"
            ));
        }

        // Assert DTR | RTS so USB CDC endpoints (RP2040 / SAMD11 / ESP32-C6) deliver host bytes.
        let mut modem_bits: c_int = TIOCM_DTR | TIOCM_RTS;
        let _ = unsafe { ioctl(fd, TIOCMBIS, &mut modem_bits as *mut c_int) };

        Ok(port)
    }

    pub fn wait_for_serial_port(port_path: &str, timeout: Duration) -> Result<SerialPort, String> {
        let deadline = Instant::now() + timeout;
        let p = Path::new(port_path);
        loop {
            if p.exists() {
                if let Ok(sp) = open_serial_115200(port_path) {
                    return Ok(sp);
                }
            }
            if Instant::now() >= deadline {
                return open_serial_115200(port_path);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    pub fn run_interactive_monitor_on(port_path: &str, port: SerialPort) -> Result<(), String> {
        eprintln!("Connected to {port_path} (115200 8N1, press Ctrl-] to exit)");
        let _tty_guard = InteractiveTtyGuard::enter(false);

        let mut fds = [
            PollFd {
                fd: STDIN_FD,
                events: POLLIN,
                revents: 0,
            },
            PollFd {
                fd: port.fd,
                events: POLLIN,
                revents: 0,
            },
        ];
        let mut buf = [0u8; 512];
        let mut stdout = io::stdout().lock();

        loop {
            fds[0].revents = 0;
            fds[1].revents = 0;
            let rc = unsafe { poll(fds.as_mut_ptr(), 2, -1) };
            if rc < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(format!("monitor poll failed: {err}"));
            }

            if (fds[1].revents & POLLIN) != 0 {
                let n = unsafe { read(port.fd, buf.as_mut_ptr().cast::<c_void>(), buf.len()) };
                if n > 0 {
                    stdout
                        .write_all(&buf[..n as usize])
                        .and_then(|()| stdout.flush())
                        .map_err(|e| format!("stdout write failed: {e}"))?;
                } else if n == 0 {
                    return Err(format!("serial device `{port_path}` disconnected"));
                }
            } else if (fds[1].revents & (POLLERR | POLLHUP)) != 0 {
                return Err(format!("serial device `{port_path}` disconnected"));
            }

            if (fds[0].revents & POLLIN) != 0 {
                let mut key = [0u8; 64];
                let n = unsafe { read(STDIN_FD, key.as_mut_ptr().cast::<c_void>(), key.len()) };
                if n <= 0 {
                    break;
                }
                let slice = &key[..n as usize];
                if let Some(pos) = slice.iter().position(|&b| b == 0x1D) {
                    if pos > 0 {
                        port.write_all_bytes(&slice[..pos])?;
                    }
                    eprintln!("\nDisconnected from {port_path}");
                    break;
                }
                port.write_all_bytes(slice)?;
            } else if (fds[0].revents & (POLLERR | POLLHUP)) != 0 {
                break;
            }
        }
        Ok(())
    }

    pub fn run_interactive_monitor(port_path: &str) -> Result<(), String> {
        let port = open_serial_115200(port_path)?;
        run_interactive_monitor_on(port_path, port)
    }
}

#[cfg(unix)]
pub use imp::{
    open_serial_115200, run_interactive_monitor, run_interactive_monitor_on, stdin_is_tty,
    wait_for_serial_port, InteractiveTtyGuard, SerialPort,
};

#[cfg(not(unix))]
pub fn stdin_is_tty() -> bool {
    false
}

#[cfg(not(unix))]
pub struct InteractiveTtyGuard;

#[cfg(not(unix))]
impl InteractiveTtyGuard {
    pub fn enter(_keep_isig: bool) -> Self {
        Self
    }
}

#[cfg(not(unix))]
pub struct SerialPort;

#[cfg(not(unix))]
impl SerialPort {
    pub fn poll_read(&self, _buf: &mut [u8], _timeout_ms: i32) -> Result<Option<usize>, String> {
        Err("serial monitor is only supported on Unix hosts".to_string())
    }

    pub fn write_all_bytes(&self, _data: &[u8]) -> Result<(), String> {
        Err("serial monitor is only supported on Unix hosts".to_string())
    }
}

#[cfg(not(unix))]
pub fn open_serial_115200(_port_path: &str) -> Result<SerialPort, String> {
    Err("serial monitor is only supported on Unix hosts".to_string())
}

#[cfg(not(unix))]
pub fn wait_for_serial_port(
    _port_path: &str,
    _timeout: std::time::Duration,
) -> Result<SerialPort, String> {
    Err("serial monitor is only supported on Unix hosts".to_string())
}

#[cfg(not(unix))]
pub fn run_interactive_monitor(_port_path: &str) -> Result<(), String> {
    Err("serial monitor is only supported on Unix hosts".to_string())
}

#[cfg(not(unix))]
pub fn run_interactive_monitor_on(_port_path: &str, _port: SerialPort) -> Result<(), String> {
    Err("serial monitor is only supported on Unix hosts".to_string())
}
