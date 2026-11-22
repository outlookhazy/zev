use std::fmt;
use zev_ast::{DurationUnit, FormatSpec, PrimType, Span};

/// A lexical token paired with its [`Span`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub const fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// Pre-tokenized segment of a zero-heap `f"..."` format string literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LexedFormatPart {
    /// Static unescaped string chunk (`{{` -> `{`, `}}` -> `}`).
    Literal(String),
    /// `{expr[:spec]}` interpolation hole with exact file-relative expression tokens.
    Hole {
        tokens: Vec<Token>,
        spec: FormatSpec,
        span: Span,
    },
}

/// Lexical token kinds for `.zev`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenKind {
    // ── Identifiers & Literals ──
    Ident(String),
    Int(u64),
    FloatLit {
        bits: u64,
        suffix: Option<PrimType>,
    },
    Duration(u64, DurationUnit),
    String(String),
    FormatString(Vec<LexedFormatPart>),
    /// Foreign Rust-style lifetime token `'a` / `'static` (trapped as `P0009`).
    Lifetime(String),

    // ── Zev Keywords ──
    KwImport,
    KwAs,
    KwPacked,
    KwStruct,
    KwEnum,
    KwConst,
    KwExtern,
    KwFn,
    KwTask,
    KwIsr,
    KwReturn,
    KwError,
    KwCatch,
    KwOr,
    KwIf,
    KwElse,
    KwWhile,
    KwFor,
    KwLoop,
    KwEvery,
    KwDefer,
    KwNoYield,
    KwCritical,
    KwSwitch,
    KwDefault,
    KwSpawn,
    KwContinue,
    KwBreak,
    KwRestart,
    KwTrue,
    KwFalse,

    // ── Foreign-Idiom Trap Keywords (`P0001`..`P0010`) ──
    KwLet,
    KwVar,
    KwMut,
    KwAsync,
    KwAwait,
    KwSelect,
    KwMatch,
    KwTry,
    KwStatic,
    KwPub,

    // ── Delimiters & Punctuation ──
    LParen,     // (
    RParen,     // )
    LBrace,     // {
    RBrace,     // }
    LBracket,   // [
    RBracket,   // ]
    Semi,       // ;
    Comma,      // ,
    Colon,      // :
    ColonColon, // ::
    Dot,        // .
    DotDot,     // ..
    DotDotEq,   // ..=
    DotDotDot,  // ...
    At,         // @
    Question,   // ?
    Arrow,      // ->
    FatArrow,   // =>

    // ── Operators ──
    Plus,       // +
    PlusPlus,   // ++
    PlusEq,     // +=
    Minus,      // -
    MinusMinus, // --
    MinusEq,    // -=
    Star,       // *
    StarEq,     // *=
    Slash,      // /
    SlashEq,    // /=
    Percent,    // %
    PercentEq,  // %=
    Amp,        // &
    AmpAmp,     // &&
    AmpEq,      // &=
    Pipe,       // |
    PipePipe,   // ||
    PipeEq,     // |=
    Caret,      // ^
    CaretEq,    // ^=
    Bang,       // !
    BangEq,     // !=
    Tilde,      // ~
    Eq,         // =
    EqEq,       // ==
    Lt,         // <
    LtEq,       // <=
    Shl,        // <<
    ShlEq,      // <<=
    Gt,         // >
    GtEq,       // >=
    Shr,        // >>
    ShrEq,      // >>=

    /// End of file.
    Eof,
}

impl TokenKind {
    pub fn keyword_from_str(s: &str) -> Option<Self> {
        match s {
            "import" => Some(Self::KwImport),
            "as" => Some(Self::KwAs),
            "packed" => Some(Self::KwPacked),
            "struct" => Some(Self::KwStruct),
            "enum" => Some(Self::KwEnum),
            "const" => Some(Self::KwConst),
            "extern" => Some(Self::KwExtern),
            "fn" => Some(Self::KwFn),
            "task" => Some(Self::KwTask),
            "isr" => Some(Self::KwIsr),
            "return" => Some(Self::KwReturn),
            "error" => Some(Self::KwError),
            "catch" => Some(Self::KwCatch),
            "or" => Some(Self::KwOr),
            "if" => Some(Self::KwIf),
            "else" => Some(Self::KwElse),
            "while" => Some(Self::KwWhile),
            "for" => Some(Self::KwFor),
            "loop" => Some(Self::KwLoop),
            "every" => Some(Self::KwEvery),
            "defer" => Some(Self::KwDefer),
            "noyield" => Some(Self::KwNoYield),
            "critical" => Some(Self::KwCritical),
            "switch" => Some(Self::KwSwitch),
            "default" => Some(Self::KwDefault),
            "spawn" => Some(Self::KwSpawn),
            "continue" => Some(Self::KwContinue),
            "break" => Some(Self::KwBreak),
            "restart" => Some(Self::KwRestart),
            "true" => Some(Self::KwTrue),
            "false" => Some(Self::KwFalse),
            // Foreign-idiom trap keywords
            "let" => Some(Self::KwLet),
            "var" => Some(Self::KwVar),
            "mut" => Some(Self::KwMut),
            "async" => Some(Self::KwAsync),
            "await" => Some(Self::KwAwait),
            "select" => Some(Self::KwSelect),
            "match" => Some(Self::KwMatch),
            "try" => Some(Self::KwTry),
            "static" => Some(Self::KwStatic),
            "pub" => Some(Self::KwPub),
            _ => None,
        }
    }

    /// If this token is a keyword or identifier, return its textual name
    /// (useful after `@` where `@critical` uses keyword `critical`, or after `.`).
    pub fn as_word(&self) -> Option<&str> {
        match self {
            Self::Ident(s) => Some(s.as_str()),
            Self::KwImport => Some("import"),
            Self::KwAs => Some("as"),
            Self::KwPacked => Some("packed"),
            Self::KwStruct => Some("struct"),
            Self::KwEnum => Some("enum"),
            Self::KwConst => Some("const"),
            Self::KwExtern => Some("extern"),
            Self::KwFn => Some("fn"),
            Self::KwTask => Some("task"),
            Self::KwIsr => Some("isr"),
            Self::KwReturn => Some("return"),
            Self::KwError => Some("error"),
            Self::KwCatch => Some("catch"),
            Self::KwOr => Some("or"),
            Self::KwIf => Some("if"),
            Self::KwElse => Some("else"),
            Self::KwWhile => Some("while"),
            Self::KwFor => Some("for"),
            Self::KwLoop => Some("loop"),
            Self::KwEvery => Some("every"),
            Self::KwDefer => Some("defer"),
            Self::KwNoYield => Some("noyield"),
            Self::KwCritical => Some("critical"),
            Self::KwSwitch => Some("switch"),
            Self::KwDefault => Some("default"),
            Self::KwSpawn => Some("spawn"),
            Self::KwContinue => Some("continue"),
            Self::KwBreak => Some("break"),
            Self::KwRestart => Some("restart"),
            Self::KwTrue => Some("true"),
            Self::KwFalse => Some("false"),
            Self::KwLet => Some("let"),
            Self::KwVar => Some("var"),
            Self::KwMut => Some("mut"),
            Self::KwAsync => Some("async"),
            Self::KwAwait => Some("await"),
            Self::KwSelect => Some("select"),
            Self::KwMatch => Some("match"),
            Self::KwTry => Some("try"),
            Self::KwStatic => Some("static"),
            Self::KwPub => Some("pub"),
            _ => None,
        }
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ident(s) => write!(f, "identifier `{s}`"),
            Self::Int(n) => write!(f, "integer `{n}`"),
            Self::FloatLit { bits, suffix } => {
                let val = f64::from_bits(*bits);
                match suffix {
                    Some(s) => write!(f, "float `{val}{}`", s.as_str()),
                    None => write!(f, "float `{val}`"),
                }
            }
            Self::Duration(n, u) => write!(f, "duration `{n}{}`", u.as_str()),
            Self::String(s) => write!(f, "string literal {s:?}"),
            Self::FormatString(_) => f.write_str("format string `f\"...\"`"),
            Self::Lifetime(lt) => write!(f, "lifetime `'{lt}`"),
            Self::LParen => f.write_str("`(`"),
            Self::RParen => f.write_str("`)`"),
            Self::LBrace => f.write_str("`{`"),
            Self::RBrace => f.write_str("`}`"),
            Self::LBracket => f.write_str("`[`"),
            Self::RBracket => f.write_str("`]`"),
            Self::Semi => f.write_str("`;`"),
            Self::Comma => f.write_str("`,`"),
            Self::Colon => f.write_str("`:`"),
            Self::ColonColon => f.write_str("`::`"),
            Self::Dot => f.write_str("`.`"),
            Self::DotDot => f.write_str("`..`"),
            Self::DotDotEq => f.write_str("`..=`"),
            Self::DotDotDot => f.write_str("`...`"),
            Self::At => f.write_str("`@`"),
            Self::Question => f.write_str("`?`"),
            Self::Arrow => f.write_str("`->`"),
            Self::FatArrow => f.write_str("`=>`"),
            Self::Plus => f.write_str("`+`"),
            Self::PlusPlus => f.write_str("`++`"),
            Self::PlusEq => f.write_str("`+=`"),
            Self::Minus => f.write_str("`-`"),
            Self::MinusMinus => f.write_str("`--`"),
            Self::MinusEq => f.write_str("`-=`"),
            Self::Star => f.write_str("`*`"),
            Self::StarEq => f.write_str("`*=`"),
            Self::Slash => f.write_str("`/`"),
            Self::SlashEq => f.write_str("`/=`"),
            Self::Percent => f.write_str("`%`"),
            Self::PercentEq => f.write_str("`%=`"),
            Self::Amp => f.write_str("`&`"),
            Self::AmpAmp => f.write_str("`&&`"),
            Self::AmpEq => f.write_str("`&=`"),
            Self::Pipe => f.write_str("`|`"),
            Self::PipePipe => f.write_str("`||`"),
            Self::PipeEq => f.write_str("`|=`"),
            Self::Caret => f.write_str("`^`"),
            Self::CaretEq => f.write_str("`^=`"),
            Self::Bang => f.write_str("`!`"),
            Self::BangEq => f.write_str("`!=`"),
            Self::Tilde => f.write_str("`~`"),
            Self::Eq => f.write_str("`=`"),
            Self::EqEq => f.write_str("`==`"),
            Self::Lt => f.write_str("`<`"),
            Self::LtEq => f.write_str("`<=`"),
            Self::Shl => f.write_str("`<<`"),
            Self::ShlEq => f.write_str("`<<=`"),
            Self::Gt => f.write_str("`>`"),
            Self::GtEq => f.write_str("`>=`"),
            Self::Shr => f.write_str("`>>`"),
            Self::ShrEq => f.write_str("`>>=`"),
            Self::Eof => f.write_str("end of file"),
            other => {
                if let Some(w) = other.as_word() {
                    write!(f, "keyword `{w}`")
                } else {
                    write!(f, "{other:?}")
                }
            }
        }
    }
}
