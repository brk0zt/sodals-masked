//
// DFA Lexer + Recursive Descent Parser → CST (Continuous Semantic Topology)
//
// Grammar (EBNF):
//   <qmem_program>    ::= { <qmem_statement> }
//   <qmem_statement>  ::= <entity> <clifford_relation> <entity> <physics_metadata> ";"
//   <entity>          ::= <string_literal>
//   <clifford_relation> ::= "CAUSES" | "MEASURES" | "CORRELATES_PHASE" | "RESONATES_WITH"
//   <physics_metadata> ::= "[" <f32_energy> "," <f32_phase_theta> "]"
//
// Design:
//   - Lexer is a strict DFA (no regex). Iterates char-by-char with state transitions.
//   - Parser is recursive descent, enforcing the grammar strictly.
//   - Output is a CST graph (not a tree), natively supporting cycles.
//   - All errors are hard compile errors with position info.
//   src/compiler/qmem_compiler.rs — QYPTOS Memory Conceptual Compiler

use std::collections::HashMap;
use std::fmt;
use num_complex::Complex64;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::f64::consts::PI;

/// PHASE 60.1: The Phasor Forge (Vector Symbolic Architecture)
/// Generates a deterministic 4096-D Complex64 hypervector for any concept.
/// Uses Xorshift64* mapped to [0, 2π) to project the concept onto a unit circle.
pub fn generate_holographic_signature(concept: &str, dim: usize) -> Vec<Complex64> {
    let mut hasher = DefaultHasher::new();
    concept.hash(&mut hasher);
    let mut state = hasher.finish();
    if state == 0 { state = 0x1234567890ABCDEF; } // Seed guard

    let mut signature = Vec::with_capacity(dim);
    for _ in 0..dim {
        // Xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let random_val = state.wrapping_mul(0x2545F4914F6CDD1D);
        
        // Map to [0, 2π) phase angle
        let theta = (random_val as f64 / u64::MAX as f64) * 2.0 * PI;
        
        // Euler's formula: e^(iθ) = cos(θ) + i*sin(θ)
        signature.push(Complex64::new(theta.cos(), theta.sin()));
    }
    signature
}

// ════════════════════════════════════════════════════════════════════════════════
// §2  TOKENS (Terminal Symbols)
// ════════════════════════════════════════════════════════════════════════════════

/// Clifford relation types — the four fundamental topological connectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationType {
    /// A → B causal chain
    Causes,
    /// A observes/collapses B
    Measures,
    /// A and B share a phase correlation (Euler: e^{iθ})
    CorrelatesPhase,
    /// A and B are in wave resonance (Clifford fibration)
    ResonatesWith,
    /// A led to positive reward B
    RewardObserved,
    /// A led to failure/collision B
    TransitionFailed,
    /// Rule discovered by program synthesis
    RuleConfirmed,
}

impl fmt::Display for RelationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelationType::Causes => write!(f, "CAUSES"),
            RelationType::Measures => write!(f, "MEASURES"),
            RelationType::CorrelatesPhase => write!(f, "CORRELATES_PHASE"),
            RelationType::ResonatesWith => write!(f, "RESONATES_WITH"),
            RelationType::RewardObserved => write!(f, "REWARD_OBSERVED"),
            RelationType::TransitionFailed => write!(f, "TRANSITION_FAILED"),
            RelationType::RuleConfirmed => write!(f, "RULE_CONFIRMED"),
        }
    }
}

/// Terminal symbols produced by the DFA lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Quoted string literal: `"concept_name"`
    Entity(String),
    /// One of the four Clifford relations
    Relation(RelationType),
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// Floating-point literal (energy or phase)
    Float(f32),
    /// Keyword: AXIOM
    Axiom,
    /// Keyword: RULE
    Rule,
    /// Keyword: DEFINES
    Defines,
    /// Keyword: COMPOSE
    Compose,
    /// Keyword: AND
    And,
    /// Operator: ->
    Arrow,
    /// Keyword: GOAL — declares a goal state with preconditions
    Goal,
    /// Keyword: REQUIRES — specifies a precondition for a GOAL
    Requires,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Entity(s) => write!(f, "\"{}\"", s),
            Token::Relation(r) => write!(f, "{}", r),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Comma => write!(f, ","),
            Token::Semicolon => write!(f, ";"),
            Token::Float(v) => write!(f, "{}", v),
            Token::Axiom => write!(f, "AXIOM"),
            Token::Rule => write!(f, "RULE"),
            Token::Defines => write!(f, "DEFINES"),
            Token::Compose => write!(f, "COMPOSE"),
            Token::And => write!(f, "AND"),
            Token::Arrow => write!(f, "->"),
            Token::Goal => write!(f, "GOAL"),
            Token::Requires => write!(f, "REQUIRES"),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// §2  COMPILE ERROR
// ════════════════════════════════════════════════════════════════════════════════

/// Hard compile error with source position.
#[derive(Debug, Clone)]
pub struct CompileError {
    pub line: usize,
    pub col: usize,
    pub message: String,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QMEM Compile Error [{}:{}]: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for CompileError {}

// ════════════════════════════════════════════════════════════════════════════════
// §3  DFA LEXER (No Regex — Pure State Machine)
// ════════════════════════════════════════════════════════════════════════════════

/// DFA states for the lexer.
#[derive(Debug, Clone, Copy, PartialEq)]
enum _LexState {
    /// Scanning whitespace / searching for next token start
    Start,
    /// Inside a `"..."` string literal
    InString,
    /// Accumulating an identifier (keyword candidate)
    InIdent,
    /// Accumulating a numeric literal (including sign, decimal point)
    InNumber,
    /// Inside a `//` line comment
    InComment,
}

/// DFA-based lexer. Iterates char-by-char with explicit state transitions.
pub struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    /// Current source position (for error reporting).
    fn loc(&self) -> (usize, usize) {
        (self.line, self.col)
    }

    /// Peek at current char without consuming.
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    /// Consume current char and advance position counters.
    fn advance(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied()?;
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    /// Skip whitespace (spaces, tabs, newlines, carriage returns).
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_ascii_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Skip `// ...` line comments.
    fn skip_line_comment(&mut self) {
        while let Some(ch) = self.advance() {
            if ch == '\n' { break; }
        }
    }

    /// Lex a quoted string literal: `"content"`.
    /// DFA state: InString. Supports `\"` escape.
    fn lex_string(&mut self) -> Result<Token, CompileError> {
        let (start_line, start_col) = self.loc();
        // Opening `"` already consumed by caller.
        let mut buf = String::new();
        loop {
            match self.advance() {
                Some('\\') => {
                    // Escape sequence
                    match self.advance() {
                        Some('"') => buf.push('"'),
                        Some('\\') => buf.push('\\'),
                        Some('n') => buf.push('\n'),
                        Some('t') => buf.push('\t'),
                        Some(other) => buf.push(other),
                        None => return Err(CompileError {
                            line: start_line, col: start_col,
                            message: "Unterminated string literal (EOF after backslash)".into(),
                        }),
                    }
                }
                Some('"') => return Ok(Token::Entity(buf)),
                Some(ch) => buf.push(ch),
                None => return Err(CompileError {
                    line: start_line, col: start_col,
                    message: "Unterminated string literal (missing closing \")".into(),
                }),
            }
        }
    }

    /// Lex an identifier (keyword candidate).
    /// DFA state: InIdent. Accumulates [A-Za-z_] chars.
    fn lex_ident(&mut self, first: char) -> Result<Token, CompileError> {
        let (start_line, start_col) = (self.line, self.col - 1); // first already consumed
        let mut buf = String::new();
        buf.push(first);
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                buf.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        // Match against the Clifford relation keywords.
        match buf.as_str() {
            "CAUSES"            => Ok(Token::Relation(RelationType::Causes)),
            "MEASURES"          => Ok(Token::Relation(RelationType::Measures)),
            "CORRELATES_PHASE"  => Ok(Token::Relation(RelationType::CorrelatesPhase)),
            "RESONATES_WITH"    => Ok(Token::Relation(RelationType::ResonatesWith)),
            "REWARD_OBSERVED"   => Ok(Token::Relation(RelationType::RewardObserved)),
            "TRANSITION_FAILED" => Ok(Token::Relation(RelationType::TransitionFailed)),
            "RULE_CONFIRMED"    => Ok(Token::Relation(RelationType::RuleConfirmed)),
            "AXIOM"             => Ok(Token::Axiom),
            "RULE"              => Ok(Token::Rule),
            "DEFINES"           => Ok(Token::Defines),
            "COMPOSE"           => Ok(Token::Compose),
            "AND"               => Ok(Token::And),
            "GOAL"              => Ok(Token::Goal),
            "REQUIRES"          => Ok(Token::Requires),
            _ => Err(CompileError {
                line: start_line, col: start_col,
                message: format!("Unknown keyword '{}'. Expected CAUSES | MEASURES | CORRELATES_PHASE | RESONATES_WITH | REWARD_OBSERVED | TRANSITION_FAILED | RULE_CONFIRMED | AXIOM | RULE | DEFINES | COMPOSE | AND | GOAL | REQUIRES", buf),
            }),
        }
    }

    /// Lex a numeric literal (f32).
    /// DFA state: InNumber. Accepts optional leading `-`, digits, optional `.digits`.
    fn lex_number(&mut self, first: char) -> Result<Token, CompileError> {
        let (start_line, start_col) = (self.line, self.col - 1);
        let mut buf = String::new();
        buf.push(first);

        // Integer part (remaining digits)
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                buf.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        // Decimal part
        if self.peek() == Some('.') {
            buf.push('.');
            self.advance();
            // Must have at least one digit after dot
            let mut has_frac = false;
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    buf.push(ch);
                    self.advance();
                    has_frac = true;
                } else {
                    break;
                }
            }
            if !has_frac {
                return Err(CompileError {
                    line: start_line, col: start_col,
                    message: format!("Invalid float literal '{}' (no digits after decimal point)", buf),
                });
            }
        }

        // Optional exponent (e/E)
        if let Some('e') | Some('E') = self.peek() {
            buf.push(self.advance().unwrap());
            if let Some('+') | Some('-') = self.peek() {
                buf.push(self.advance().unwrap());
            }
            let mut has_exp = false;
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    buf.push(ch);
                    self.advance();
                    has_exp = true;
                } else {
                    break;
                }
            }
            if !has_exp {
                return Err(CompileError {
                    line: start_line, col: start_col,
                    message: format!("Invalid float literal '{}' (no digits in exponent)", buf),
                });
            }
        }

        buf.parse::<f32>().map(Token::Float).map_err(|e| CompileError {
            line: start_line, col: start_col,
            message: format!("Cannot parse float '{}': {}", buf, e),
        })
    }

    /// Main DFA dispatch. Returns `None` at EOF, `Some(Ok(token))` on success,
    /// `Some(Err(CompileError))` on lexical error.
    pub fn next_token(&mut self) -> Option<Result<Token, CompileError>> {
        loop {
            self.skip_whitespace();
            let ch = self.peek()?;

            // Single-char tokens (immediate transitions)
            match ch {
                '[' => { self.advance(); return Some(Ok(Token::LBracket)); }
                ']' => { self.advance(); return Some(Ok(Token::RBracket)); }
                ',' => { self.advance(); return Some(Ok(Token::Comma)); }
                ';' => { self.advance(); return Some(Ok(Token::Semicolon)); }
                '"' => {
                    self.advance(); // consume opening quote
                    return Some(self.lex_string());
                }
                '/' => {
                    // Check for line comment
                    self.advance();
                    if self.peek() == Some('/') {
                        self.skip_line_comment();
                        continue; // Restart DFA after comment
                    } else {
                        return Some(Err(CompileError {
                            line: self.line, col: self.col - 1,
                            message: "Unexpected '/' (did you mean '//' for a comment?)".into(),
                        }));
                    }
                }
                '-' => {
                    // Negative number or arrow
                    self.advance();
                    if self.peek() == Some('>') {
                        self.advance();
                        return Some(Ok(Token::Arrow));
                    }
                    if let Some(d) = self.peek() {
                        if d.is_ascii_digit() {
                            return Some(self.lex_number('-'));
                        }
                    }
                    return Some(Err(CompileError {
                        line: self.line, col: self.col - 1,
                        message: "Unexpected '-' (expected '>' for arrow or digit after minus sign)".into(),
                    }));
                }
                _ if ch.is_ascii_digit() => {
                    self.advance();
                    return Some(self.lex_number(ch));
                }
                _ if ch.is_ascii_alphabetic() || ch == '_' => {
                    self.advance();
                    return Some(self.lex_ident(ch));
                }
                _ => {
                    let (l, c) = self.loc();
                    self.advance();
                    return Some(Err(CompileError {
                        line: l, col: c,
                        message: format!("Unexpected character '{}'", ch),
                    }));
                }
            }
        }
    }

    /// Drain all tokens into a Vec (for debugging / batch processing).
    pub fn tokenize_all(&mut self) -> Result<Vec<Token>, CompileError> {
        let mut tokens = Vec::new();
        while let Some(result) = self.next_token() {
            tokens.push(result?);
        }
        Ok(tokens)
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// §4  CST (Continuous Semantic Topology) — A Graph, Not a Tree
// ════════════════════════════════════════════════════════════════════════════════

/// Directed edge in the CST carrying Clifford physics metadata.
#[derive(Debug, Clone)]
pub struct SemanticEdge {
    /// Target concept (node key in the topology's HashMap)
    pub target: String,
    /// Clifford relation type
    pub relation: RelationType,
    /// Diffusion energy scalar [0, ∞)
    pub energy: f32,
    /// Euler phase angle θ ∈ (-π, π]
    pub phase_theta: f32,
}

/// Node in the CST. Holds outgoing edges (adjacency list).
/// Cycles are natively supported: node A can have an edge → B and B → A.
#[derive(Debug, Clone)]
pub struct CstNode {
    /// Canonical concept name
    pub concept: String,
    /// Outgoing semantic edges
    pub edges: Vec<SemanticEdge>,
    // PHASE 60.1: Holographic Hyperdimensional Signature
    pub holographic_signature: Vec<Complex64>,
}

impl CstNode {
    fn new(concept: String) -> Self {
        Self {
            concept: concept.clone(),
            edges: Vec::new(),
            holographic_signature: generate_holographic_signature(&concept, 4096),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AxiomNode {
    pub name: String,
    pub concept: String,
    pub energy: f32,
    pub phase_theta: f32,
}

#[derive(Debug, Clone)]
pub struct CompositionRule {
    pub name: String,
    pub concept_a: String,
    pub concept_b: String,
    pub target_concept: String,
    pub energy: f32,
    pub phase_theta: f32,
}

/// Goal-precondition node: GOAL "goal_state" REQUIRES "precondition_state" [energy, phase];
///
/// Models the STRIPS/PDDL-style reasoning: "to achieve goal G, precondition P must be satisfied."
/// This is fundamentally different from analogy/coherence reasoning —
/// it answers "does this action satisfy the goal's physical prerequisites?"
/// rather than "is this action semantically consistent?"
#[derive(Debug, Clone)]
pub struct GoalNode {
    /// The desired goal state (e.g., "araba_yikanmis")
    pub goal_state: String,
    /// The required precondition (e.g., "araba_konumu == yikama_yeri")
    pub precondition: String,
    /// Energy weight of this goal-precondition link
    pub energy: f32,
    /// Phase angle of this goal-precondition link
    pub phase_theta: f32,
}

/// The CST — a directed graph of concepts connected by typed, physics-annotated edges.
///
/// This is NOT an AST. Trees cannot represent cycles.
/// The CST is the compiler's intermediate representation that maps directly
/// onto `ResonanceNetwork` for physics simulation.
#[derive(Debug, Clone)]
pub struct ContinuousSemanticTopology {
    /// concept_name → CstNode. Insertion-order is irrelevant (topology is the truth).
    pub nodes: HashMap<String, CstNode>,
    pub axioms: Vec<AxiomNode>,
    pub rules: Vec<CompositionRule>,
    /// Goal-precondition constraints for STRIPS-style reasoning
    pub goals: Vec<GoalNode>,
}

impl Default for ContinuousSemanticTopology {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinuousSemanticTopology {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            axioms: Vec::new(),
            rules: Vec::new(),
            goals: Vec::new(),
        }
    }

    /// Ensure a node exists for the given concept. Returns mutable ref.
    fn ensure_node(&mut self, concept: &str) -> &mut CstNode {
        self.nodes.entry(concept.to_string())
            .or_insert_with(|| CstNode::new(concept.to_string()))
    }

    /// Insert a parsed statement into the topology.
    fn insert_edge(
        &mut self,
        source: &str,
        target: &str,
        relation: RelationType,
        energy: f32,
        phase_theta: f32,
    ) {
        // Ensure both nodes exist
        self.ensure_node(target);
        let src = self.ensure_node(source);
        src.edges.push(SemanticEdge {
            target: target.to_string(),
            relation,
            energy,
            phase_theta,
        });
    }

    /// Total directed edges in the CST.
    pub fn edge_count(&self) -> usize {
        self.nodes.values().map(|n| n.edges.len()).sum()
    }

    /// Total unique concepts.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl fmt::Display for ContinuousSemanticTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CST ({} nodes, {} edges, {} axioms, {} rules, {} goals):", self.node_count(), self.edge_count(), self.axioms.len(), self.rules.len(), self.goals.len())?;
        for (name, node) in &self.nodes {
            for edge in &node.edges {
                writeln!(f, "  \"{}\" {} \"{}\" [E={:.4}, θ={:.4}]",
                    name, edge.relation, edge.target, edge.energy, edge.phase_theta)?;
            }
        }
        for axiom in &self.axioms {
            writeln!(f, "  AXIOM \"{}\" DEFINES \"{}\" [E={:.4}, θ={:.4}]",
                axiom.name, axiom.concept, axiom.energy, axiom.phase_theta)?;
        }
        for rule in &self.rules {
            writeln!(f, "  RULE \"{}\" COMPOSE \"{}\" AND \"{}\" -> \"{}\" [E={:.4}, θ={:.4}]",
                rule.name, rule.concept_a, rule.concept_b, rule.target_concept, rule.energy, rule.phase_theta)?;
        }
        for goal in &self.goals {
            writeln!(f, "  GOAL \"{}\" REQUIRES \"{}\" [E={:.4}, θ={:.4}]",
                goal.goal_state, goal.precondition, goal.energy, goal.phase_theta)?;
        }
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// §5  RECURSIVE DESCENT PARSER
// ════════════════════════════════════════════════════════════════════════════════

/// Recursive descent parser consuming tokens from the `Lexer`.
/// Enforces the QMEM grammar strictly. Hard errors on malformed input.
pub struct Parser {
    lexer: Lexer,
    /// Lookahead (1-token buffer for LL(1) parsing)
    current: Option<Token>,
    line: usize,
    col: usize,
}

impl Parser {
    pub fn new(input: &str) -> Self {
        let mut lexer = Lexer::new(input);
        // Prime the lookahead
        let (line, col) = lexer.loc();
        let current = match lexer.next_token() {
            Some(Ok(t)) => Some(t),
            Some(Err(_)) => None, // Will be caught on first parse call
            None => None,
        };
        Self { lexer, current, line, col }
    }

    /// Advance to the next token.
    fn bump(&mut self) -> Result<(), CompileError> {
        self.line = self.lexer.line;
        self.col = self.lexer.col;
        self.current = match self.lexer.next_token() {
            Some(Ok(t)) => Some(t),
            Some(Err(e)) => return Err(e),
            None => None,
        };
        Ok(())
    }

    /// Expect and consume a specific token type. Hard error on mismatch.
    fn expect_entity(&mut self) -> Result<String, CompileError> {
        match self.current.take() {
            Some(Token::Entity(s)) => { self.bump()?; Ok(s) }
            other => Err(CompileError {
                line: self.line, col: self.col,
                message: format!("Expected string literal (entity), got {:?}", other),
            }),
        }
    }

    fn expect_relation(&mut self) -> Result<RelationType, CompileError> {
        match self.current.take() {
            Some(Token::Relation(r)) => { self.bump()?; Ok(r) }
            other => Err(CompileError {
                line: self.line, col: self.col,
                message: format!("Expected relation (CAUSES|MEASURES|CORRELATES_PHASE|RESONATES_WITH|REWARD_OBSERVED|TRANSITION_FAILED|RULE_CONFIRMED), got {:?}", other),
            }),
        }
    }

    fn expect_float(&mut self) -> Result<f32, CompileError> {
        match self.current.take() {
            Some(Token::Float(v)) => { self.bump()?; Ok(v) }
            other => Err(CompileError {
                line: self.line, col: self.col,
                message: format!("Expected float literal, got {:?}", other),
            }),
        }
    }

    fn expect_token(&mut self, expected: Token) -> Result<(), CompileError> {
        match self.current.take() {
            Some(ref t) if *t == expected => { self.bump()?; Ok(()) }
            other => Err(CompileError {
                line: self.line, col: self.col,
                message: format!("Expected '{}', got {:?}", expected, other),
            }),
        }
    }

    /// Parse: `<physics_metadata> ::= "[" <f32_energy> "," <f32_phase_theta> "]"`
    fn parse_physics_metadata(&mut self) -> Result<(f32, f32), CompileError> {
        self.expect_token(Token::LBracket)?;
        let energy = self.expect_float()?;
        self.expect_token(Token::Comma)?;
        let phase_theta = self.expect_float()?;
        self.expect_token(Token::RBracket)?;
        Ok((energy, phase_theta))
    }

    /// Parse: `<qmem_statement> ::= <relation_statement> | <axiom_statement> | <rule_statement> | <goal_statement>`
    ///
    /// Goal syntax: GOAL "goal_state" REQUIRES "precondition" [energy, phase];
    fn parse_statement(&mut self, cst: &mut ContinuousSemanticTopology) -> Result<(), CompileError> {
        match &self.current {
            Some(Token::Axiom) => {
                self.expect_token(Token::Axiom)?;
                let name = self.expect_entity()?;
                self.expect_token(Token::Defines)?;
                let concept = self.expect_entity()?;
                let (energy, phase_theta) = self.parse_physics_metadata()?;
                self.expect_token(Token::Semicolon)?;
                
                cst.axioms.push(AxiomNode {
                    name,
                    concept,
                    energy,
                    phase_theta,
                });
                Ok(())
            }
            Some(Token::Rule) => {
                self.expect_token(Token::Rule)?;
                let name = self.expect_entity()?;
                self.expect_token(Token::Compose)?;
                let concept_a = self.expect_entity()?;
                self.expect_token(Token::And)?;
                let concept_b = self.expect_entity()?;
                self.expect_token(Token::Arrow)?;
                let target_concept = self.expect_entity()?;
                let (energy, phase_theta) = self.parse_physics_metadata()?;
                self.expect_token(Token::Semicolon)?;
                
                cst.rules.push(CompositionRule {
                    name,
                    concept_a,
                    concept_b,
                    target_concept,
                    energy,
                    phase_theta,
                });
                Ok(())
            }
            Some(Token::Goal) => {
                // GOAL "goal_state" REQUIRES "precondition" [energy, phase];
                self.expect_token(Token::Goal)?;
                let goal_state = self.expect_entity()?;
                self.expect_token(Token::Requires)?;
                let precondition = self.expect_entity()?;
                let (energy, phase_theta) = self.parse_physics_metadata()?;
                self.expect_token(Token::Semicolon)?;

                cst.goals.push(GoalNode {
                    goal_state,
                    precondition,
                    energy,
                    phase_theta,
                });
                Ok(())
            }
            _ => {
                let source = self.expect_entity()?;
                let relation = self.expect_relation()?;
                let target = self.expect_entity()?;
                let (energy, phase_theta) = self.parse_physics_metadata()?;
                self.expect_token(Token::Semicolon)?;

                cst.insert_edge(&source, &target, relation, energy, phase_theta);
                Ok(())
            }
        }
    }

    /// Parse: `<qmem_program> ::= { <qmem_statement> }`
    ///
    /// Returns the fully constructed CST graph.
    pub fn parse(&mut self) -> Result<ContinuousSemanticTopology, CompileError> {
        let mut cst = ContinuousSemanticTopology::new();
        let mut statement_count = 0_usize;

        while self.current.is_some() {
            self.parse_statement(&mut cst)?;
            statement_count += 1;
        }

        if statement_count == 0 {
            return Err(CompileError {
                line: 1, col: 1,
                message: "Empty QMEM program (no statements found)".into(),
            });
        }

        eprintln!(
            "[QMEM COMPILER] Compiled {} statements → CST ({} nodes, {} edges)",
            statement_count, cst.node_count(), cst.edge_count()
        );

        Ok(cst)
    }
}

/// Convenience function: compile a QMEM source string into a CST in one call.
pub fn compile(source: &str) -> Result<ContinuousSemanticTopology, CompileError> {
    Parser::new(source).parse()
}

// ════════════════════════════════════════════════════════════════════════════════
// §6  TEST SUITE
// ════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Lexer Tests ───────────────────────────────────────────────

    #[test]
    fn test_lexer_single_statement() {
        let input = r#""gravity" CAUSES "mass" [0.95, 3.14];"#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize_all().unwrap();
        assert_eq!(tokens.len(), 9);
        assert_eq!(tokens[0], Token::Entity("gravity".into()));
        assert_eq!(tokens[1], Token::Relation(RelationType::Causes));
        assert_eq!(tokens[2], Token::Entity("mass".into()));
        assert_eq!(tokens[3], Token::LBracket);
        assert!(matches!(tokens[4], Token::Float(v) if (v - 0.95).abs() < 1e-6));
        assert_eq!(tokens[5], Token::Comma);
        assert!(matches!(tokens[6], Token::Float(v) if (v - 3.14).abs() < 1e-2));
        assert_eq!(tokens[7], Token::RBracket);
        assert_eq!(tokens[8], Token::Semicolon);
    }

    #[test]
    fn test_lexer_all_relations() {
        let input = "CAUSES MEASURES CORRELATES_PHASE RESONATES_WITH";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize_all().unwrap();
        assert_eq!(tokens, vec![
            Token::Relation(RelationType::Causes),
            Token::Relation(RelationType::Measures),
            Token::Relation(RelationType::CorrelatesPhase),
            Token::Relation(RelationType::ResonatesWith),
        ]);
    }

    #[test]
    fn test_lexer_negative_float() {
        let input = "[-0.5, -3.14]";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize_all().unwrap();
        assert_eq!(tokens.len(), 5);
        assert!(matches!(tokens[1], Token::Float(v) if (v - (-0.5)).abs() < 1e-6));
        assert!(matches!(tokens[3], Token::Float(v) if (v - (-3.14)).abs() < 1e-2));
    }

    #[test]
    fn test_lexer_comment_skipping() {
        let input = "// This is a comment\n\"a\" CAUSES \"b\" [1.0, 0.0];";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize_all().unwrap();
        assert_eq!(tokens.len(), 9); // comment produces no tokens
    }

    #[test]
    fn test_lexer_unterminated_string() {
        let input = r#""unterminated"#;
        let mut lexer = Lexer::new(input);
        let result = lexer.tokenize_all();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("Unterminated"));
    }

    #[test]
    fn test_lexer_unknown_keyword() {
        let input = "UNKNOWN_RELATION";
        let mut lexer = Lexer::new(input);
        let result = lexer.tokenize_all();
        assert!(result.is_err());
    }

    #[test]
    fn test_lexer_escape_sequences() {
        let input = r#""hello\"world""#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize_all().unwrap();
        assert_eq!(tokens[0], Token::Entity("hello\"world".into()));
    }

    // ─── Parser Tests ──────────────────────────────────────────────

    #[test]
    fn test_parser_single_statement() {
        let input = r#""gravity" CAUSES "mass" [0.95, 3.14];"#;
        let cst = compile(input).unwrap();
        assert_eq!(cst.node_count(), 2);
        assert_eq!(cst.edge_count(), 1);

        let gravity = &cst.nodes["gravity"];
        assert_eq!(gravity.edges.len(), 1);
        assert_eq!(gravity.edges[0].target, "mass");
        assert_eq!(gravity.edges[0].relation, RelationType::Causes);
        assert!((gravity.edges[0].energy - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_parser_multiple_statements() {
        let input = r#"
            "gravity" CAUSES "mass" [0.95, 3.14];
            "mass" RESONATES_WITH "energy" [0.8, 1.57];
            "energy" CORRELATES_PHASE "gravity" [0.6, -0.78];
        "#;
        let cst = compile(input).unwrap();
        assert_eq!(cst.node_count(), 3); // gravity, mass, energy
        assert_eq!(cst.edge_count(), 3);

        // Verify cycle: energy → gravity → mass → energy (topologically connected)
        assert!(cst.nodes["energy"].edges.iter().any(|e| e.target == "gravity"));
        assert!(cst.nodes["gravity"].edges.iter().any(|e| e.target == "mass"));
        assert!(cst.nodes["mass"].edges.iter().any(|e| e.target == "energy"));
    }

    #[test]
    fn test_parser_measures_relation() {
        let input = r#""observer" MEASURES "wavefunction" [1.0, 0.0];"#;
        let cst = compile(input).unwrap();
        assert_eq!(cst.nodes["observer"].edges[0].relation, RelationType::Measures);
    }

    #[test]
    fn test_parser_missing_semicolon() {
        let input = r#""a" CAUSES "b" [1.0, 0.0]"#;
        let result = compile(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("Expected ';'"));
    }

    #[test]
    fn test_parser_axiom() {
        let input = r#"AXIOM "gravity" DEFINES "mass_dependent" [0.95, 0.0];"#;
        let cst = compile(input).unwrap();
        assert_eq!(cst.axioms.len(), 1);
        assert_eq!(cst.axioms[0].name, "gravity");
        assert_eq!(cst.axioms[0].concept, "mass_dependent");
        assert_eq!(cst.axioms[0].energy, 0.95);
        assert_eq!(cst.axioms[0].phase_theta, 0.0);
    }

    #[test]
    fn test_parser_rule() {
        let input = r#"RULE "kurgusal_gravity" COMPOSE "wavelength_dependent" AND "gravity" -> "wavelength_gravity" [0.95, 0.0];"#;
        let cst = compile(input).unwrap();
        assert_eq!(cst.rules.len(), 1);
        assert_eq!(cst.rules[0].name, "kurgusal_gravity");
        assert_eq!(cst.rules[0].concept_a, "wavelength_dependent");
        assert_eq!(cst.rules[0].concept_b, "gravity");
        assert_eq!(cst.rules[0].target_concept, "wavelength_gravity");
        assert_eq!(cst.rules[0].energy, 0.95);
        assert_eq!(cst.rules[0].phase_theta, 0.0);
    }

    #[test]
    fn test_parser_missing_metadata() {
        let input = r#""a" CAUSES "b";"#;
        let result = compile(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("Expected '['"));
    }

    #[test]
    fn test_parser_empty_program() {
        let result = compile("");
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("Empty QMEM program"));
    }

    #[test]
    fn test_parser_with_comments() {
        let input = r#"
            // This defines causal gravity
            "gravity" CAUSES "mass" [0.9, 0.0];
            // End of program
        "#;
        let cst = compile(input).unwrap();
        assert_eq!(cst.node_count(), 2);
    }

    #[test]
    fn test_parser_self_loop() {
        let input = r#""consciousness" RESONATES_WITH "consciousness" [1.0, 6.28];"#;
        let cst = compile(input).unwrap();
        assert_eq!(cst.node_count(), 1); // single node with self-edge
        assert_eq!(cst.edge_count(), 1);
        assert_eq!(cst.nodes["consciousness"].edges[0].target, "consciousness");
    }

    #[test]
    fn test_parser_negative_energy() {
        let input = r#""entropy" CAUSES "decay" [-0.5, -1.57];"#;
        let cst = compile(input).unwrap();
        assert!((cst.nodes["entropy"].edges[0].energy - (-0.5)).abs() < 1e-6);
        assert!((cst.nodes["entropy"].edges[0].phase_theta - (-1.57)).abs() < 1e-2);
    }

    // ─── Display / Formatting ──────────────────────────────────────

    #[test]
    fn test_cst_display() {
        let input = r#""a" CAUSES "b" [1.0, 0.5];"#;
        let cst = compile(input).unwrap();
        let display = format!("{}", cst);
        assert!(display.contains("CST (2 nodes, 1 edges, 0 axioms, 0 rules, 0 goals)"));
        assert!(display.contains("CAUSES"));
    }

    // ─── GOAL / REQUIRES Tests ─────────────────────────────────────

    #[test]
    fn test_parser_goal_basic() {
        let input = r#"GOAL "araba_yikanmis" REQUIRES "araba_yikama_yerinde" [1.0, 0.0];"#;
        let cst = compile(input).unwrap();
        assert_eq!(cst.goals.len(), 1);
        assert_eq!(cst.goals[0].goal_state, "araba_yikanmis");
        assert_eq!(cst.goals[0].precondition, "araba_yikama_yerinde");
        assert!((cst.goals[0].energy - 1.0).abs() < 1e-6);
        assert!((cst.goals[0].phase_theta - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_parser_goal_mixed_with_axioms_and_rules() {
        let input = r#"
            // Car wash scenario
            AXIOM "arabayla_git" DEFINES "araba_konumu_yikama" [1.0, 0.0];
            AXIOM "yuruyerek_git" DEFINES "insan_konumu_yikama" [1.0, 0.0];
            "arabayla_git" CAUSES "araba_konumu_yikama" [0.95, 0.0];
            GOAL "araba_yikanmis" REQUIRES "araba_konumu_yikama" [1.0, 0.0];
        "#;
        let cst = compile(input).unwrap();
        assert_eq!(cst.axioms.len(), 2);
        assert_eq!(cst.goals.len(), 1);
        assert_eq!(cst.edge_count(), 1); // One CAUSES edge
        assert_eq!(cst.goals[0].goal_state, "araba_yikanmis");
        assert_eq!(cst.goals[0].precondition, "araba_konumu_yikama");
    }

    #[test]
    fn test_parser_multiple_goals() {
        let input = r#"
            GOAL "araba_yikanmis" REQUIRES "araba_yikama_yerinde" [1.0, 0.0];
            GOAL "insan_eve_donmus" REQUIRES "ulasim_araci_mevcut" [0.8, 0.5];
        "#;
        let cst = compile(input).unwrap();
        assert_eq!(cst.goals.len(), 2);
        assert_eq!(cst.goals[0].goal_state, "araba_yikanmis");
        assert_eq!(cst.goals[1].goal_state, "insan_eve_donmus");
    }

    #[test]
    fn test_goal_display_format() {
        let input = r#"GOAL "araba_yikanmis" REQUIRES "araba_yikama_yerinde" [1.0, 0.0];"#;
        let cst = compile(input).unwrap();
        let display = format!("{}", cst);
        assert!(display.contains("GOAL"), "Display should show GOAL keyword");
        assert!(display.contains("REQUIRES"), "Display should show REQUIRES keyword");
        assert!(display.contains("1 goals"), "Display should count goals");
    }

    #[test]
    fn test_lexer_goal_requires_tokens() {
        let input = "GOAL REQUIRES";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize_all().unwrap();
        assert_eq!(tokens, vec![Token::Goal, Token::Requires]);
    }
}
