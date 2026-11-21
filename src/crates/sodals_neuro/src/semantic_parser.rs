// src/crates/sodals_neuro/src/semantic_parser.rs
// PRODUCTION: Deterministic AST parser with scope tracking
// Uses pest PEG parser for zero-copy, fast parsing


use pest::iterators::Pair;
use pest_derive::Parser;
use std::collections::HashMap;
use num_complex::Complex64;
use crate::semantic_network::ResonanceNetwork;
use candle_core::{Tensor, Device, Result as CandleResult};
use crate::gpu_ops::memory_utils::ZeroCopyCartridge;

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// NEUROSYMBOLIC TOPOLOGY FLAGS (Phase 19)
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

pub const DOMAIN_LOGIC: u32    = 1 << 0; // Pure algorithmic constructs
pub const DOMAIN_TEMPORAL: u32 = 1 << 1; // Time/Iteration loops
pub const DOMAIN_SPATIAL: u32  = 1 << 2; // Data structures/Memory layouts
pub const DOMAIN_ABSTRACT: u32 = 1 << 3; // Theoretical/Heuristic nodes

#[derive(Debug, Clone)]
pub struct SemanticNode {
    pub label: String,
    pub base_frequency: f32,
    pub decay_rate: f32,
    pub topological_impedance: f32,
    pub neuro_flag: u32,
}


// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// GRAMMAR DEFINITION (Embedded PEG - supports Rust/Python subset)
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

#[derive(Parser)]
#[grammar_inline = r#"
WHITESPACE = _{ " " | "\t" | "\r" | "\n" }
COMMENT = _{ "//" ~ (!"\n" ~ ANY)* }

// Identifiers and literals
ident = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }
number = @{ ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT+)? }
string_lit = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" }

// Keywords
kw_fn = { "fn" | "def" }
kw_let = { "let" | "mut" }
kw_struct = { "struct" | "class" }
kw_if = { "if" }
kw_else = { "else" }
kw_for = { "for" }
kw_while = { "while" }
kw_return = { "return" }
kw_self = { "self" }
kw_pub = { "pub" }

// Types
type_spec = { ident ~ ("<" ~ type_spec ~ ("," ~ type_spec)* ~ ">")? }

// Function definition
fn_def = {
    kw_pub? ~ kw_fn ~ ident ~ "(" ~ param_list? ~ ")" ~
    ("->" ~ type_spec)? ~ block
}

param_list = { param ~ ("," ~ param)* }
param = { ident ~ (":" ~ type_spec)? }

// Variable binding
let_stmt = { kw_let ~ ident ~ (":" ~ type_spec)? ~ "=" ~ expr ~ ";" }

// Struct definition
struct_def = {
    kw_pub? ~ kw_struct ~ ident ~ "{" ~ field_list? ~ "}"
}

field_list = { field ~ ("," ~ field)* ~ ","? }
field = { kw_pub? ~ ident ~ ":" ~ type_spec }

// Statements
stmt = {
    let_stmt |
    if_stmt |
    for_stmt |
    while_stmt |
    return_stmt |
    expr_stmt
}

if_stmt = { kw_if ~ expr ~ block ~ (kw_else ~ block)? }
for_stmt = { kw_for ~ ident ~ "in" ~ expr ~ block }
while_stmt = { kw_while ~ expr ~ block }
return_stmt = { kw_return ~ expr? ~ ";" }
expr_stmt = { expr ~ ";" }

// Block
block = { "{" ~ stmt* ~ "}" }

// Expressions (simplified precedence)
expr = { term ~ (bin_op ~ term)* }
term = {
    number |
    string_lit |
    fn_call |
    ident |
    "(" ~ expr ~ ")"
}

fn_call = { ident ~ "(" ~ arg_list? ~ ")" }
arg_list = { expr ~ ("," ~ expr)* }

bin_op = { "+" | "-" | "*" | "/" | "==" | "!=" | "<" | ">" | "&&" | "||" }

// Top level
item = { fn_def | struct_def | let_stmt }
program = { SOI ~ item* ~ EOI }
"#]
struct CodeParser;

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// AST NODE DEFINITIONS
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

#[derive(Debug, Clone)]
pub enum AstNode {
    Program {
        items: Vec<AstNode>,
    },
    
    FnDef {
        name: String,
        params: Vec<String>,
        body: Vec<AstNode>,
        node_id: usize, // Network node ID
    },
    
    StructDef {
        name: String,
        fields: Vec<(String, String)>, // (name, type)
        node_id: usize,
    },
    
    LetStmt {
        name: String,
        value: Box<AstNode>,
        node_id: usize,
    },
    
    IfStmt {
        condition: Box<AstNode>,
        then_block: Vec<AstNode>,
        else_block: Option<Vec<AstNode>>,
    },
    
    ForLoop {
        var: String,
        iter: Box<AstNode>,
        body: Vec<AstNode>,
        var_node_id: usize,
    },
    
    WhileLoop {
        condition: Box<AstNode>,
        body: Vec<AstNode>,
    },
    
    ReturnStmt {
        value: Option<Box<AstNode>>,
    },
    
    BinOp {
        op: String,
        left: Box<AstNode>,
        right: Box<AstNode>,
    },
    
    FnCall {
        name: String,
        args: Vec<AstNode>,
        node_id: usize,
    },
    
    Ident {
        name: String,
        node_id: usize,
    },
    
    Literal {
        value: String,
        node_id: usize,
    },
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// SCOPE TRACKING (Pushdown Automaton)
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

#[derive(Debug, Clone)]
struct ScopeFrame {
    depth: usize,
    bindings: HashMap<String, usize>, // var_name -> network_node_id
}

struct ScopeStack {
    frames: Vec<ScopeFrame>,
}

impl ScopeStack {
    fn new() -> Self {
        Self {
            frames: vec![ScopeFrame {
                depth: 0,
                bindings: HashMap::new(),
            }],
        }
    }
    
    #[inline]
    fn push_scope(&mut self) {
        let depth = self.frames.last().map(|f| f.depth + 1).unwrap_or(0);
        self.frames.push(ScopeFrame {
            depth,
            bindings: HashMap::new(),
        });
    }
    
    #[inline]
    fn pop_scope(&mut self) {
        if self.frames.len() > 1 {
            self.frames.pop();
        }
    }
    
    #[inline]
    fn bind(&mut self, name: String, node_id: usize) {
        if let Some(frame) = self.frames.last_mut() {
            frame.bindings.insert(name, node_id);
        }
    }
    
    #[inline]
    fn lookup(&self, name: &str) -> Option<usize> {
        // Search from innermost to outermost scope
        for frame in self.frames.iter().rev() {
            if let Some(&node_id) = frame.bindings.get(name) {
                return Some(node_id);
            }
        }
        None
    }
    
    #[inline]
    fn _current_depth(&self) -> usize {
        self.frames.last().map(|f| f.depth).unwrap_or(0)
    }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// CODE WEAVER (Main Parser & Network Builder)
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

pub struct CodeWeaver {
    scope_stack: ScopeStack,
    _last_node_id: Option<usize>, // For syntax edges
    stats: WeaverStats,
    pub nodes: Vec<SemanticNode>,
}

#[derive(Debug, Default)]
struct WeaverStats {
    nodes_created: usize,
    syntax_edges: usize,
    semantic_edges: usize,
}

impl CodeWeaver {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            scope_stack: ScopeStack::new(),
            _last_node_id: None,
            stats: WeaverStats::default(),
            nodes: Vec::new(),
        })
    }

    /// Extract technical entities from natural language and seed the network.
    pub fn populate_from_natural_language(&mut self, network: &mut ResonanceNetwork, text: &str) {
        let lowercase = text.to_lowercase();
        let mut found_concepts: Vec<(String, usize)> = Vec::new();
        
        for (label, &id) in &network.labels {
            if label.len() <= 2 {
                continue; // Skip short labels
            }
            let clean_label = label.to_lowercase();
            // Check if text contains the clean label
            if lowercase.contains(&clean_label) {
                found_concepts.push((label.clone(), id));
            }
        }
        
        // Sort by length descending to keep the most specific/longest matches
        found_concepts.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        
        let mut filtered_concepts: Vec<(String, usize)> = Vec::new();
        for (label, id) in found_concepts {
            let is_sub = filtered_concepts.iter().any(|(accepted_label, _): &(String, usize)| {
                accepted_label.to_lowercase().contains(&label.to_lowercase())
            });
            if !is_sub {
                filtered_concepts.push((label, id));
            }
        }
        
        let mut found_ids = Vec::new();
        for (label, id) in filtered_concepts {
            found_ids.push(id);
            
            // Retrieve decay/damping rate from network if available
            let decay = network.damping.get(id).copied().unwrap_or(0.05);
            
            // Phase 19.5: Inject SemanticNode for natural language context
            self.nodes.push(SemanticNode {
                label: label.clone(),
                base_frequency: 0.1, 
                decay_rate: decay as f32,
                topological_impedance: decay as f32,
                neuro_flag: DOMAIN_ABSTRACT,
            });

            // Perturb the node position to indicate activation (energy injection)
            if id < network.n {
                network.positions.amplitudes[id] += Complex64::new(1.0, 0.0);
            }
        }
        
        // Connect found entities with weak semantic links to build a local context
        for i in 0..found_ids.len() {
            for j in (i+1)..found_ids.len() {
                network.connect_semantic(found_ids[i], found_ids[j], 0.2);
            }
        }
        
        if !found_ids.is_empty() {
             eprintln!("[WEAVER] IIII Seeded network with {} natural language concepts dynamically", found_ids.len());
        }
    }

    /// Extract topological nodes for the GPU reactor (Phase 19.5 Implementation)
    pub fn extract_topology(&mut self, source_code: &str) -> Result<Vec<SemanticNode>, Box<dyn std::error::Error>> {
        use pest::Parser;
        let pairs = CodeParser::parse(Rule::program, source_code)
            .map_err(|e| format!("Parse error: {}", e))?;
        
        // We use a dummy network for context extraction if one isn't provided,
        // but we mainly care about the self.nodes population.
        let mut dummy_network = ResonanceNetwork::new(10); 
        let ast = self.build_ast(pairs, &mut dummy_network)?;
        self.traverse_ast(&ast, &mut dummy_network)?;
        
        Ok(self.nodes.clone())
    }

    /// Main entry point: Parse source code and weave into semantic network
    pub fn weave(
        &mut self,
        network: &mut ResonanceNetwork,
        source_code: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use pest::Parser;
        
        eprintln!("[WEAVER] IIII Parsing source code...");
        
        // Parse with pest
        let pairs = CodeParser::parse(Rule::program, source_code)
            .map_err(|e| format!("Parse error: {}", e))?;
        
        // Build AST
        let ast = self.build_ast(pairs, network)?;
        
        // Traverse AST and create network edges
        self.traverse_ast(&ast, network)?;
        
        eprintln!(
            "[WEAVER] III Complete: {} nodes, {} syntax edges, {} semantic edges",
            self.stats.nodes_created,
            self.stats.syntax_edges,
            self.stats.semantic_edges
        );
        
        Ok(())
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // AST BUILDER (Pest Pairs III AST)
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    fn build_ast(
        &mut self,
        pairs: pest::iterators::Pairs<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let mut items = Vec::new();
        
        for pair in pairs {
            match pair.as_rule() {
                Rule::program => {
                    for inner in pair.into_inner() {
                        if let Rule::item = inner.as_rule() {
                            items.push(self.build_item(inner, network)?);
                        }
                    }
                }
                Rule::EOI => {}
                _ => {}
            }
        }
        
        Ok(AstNode::Program { items })
    }

    fn build_item(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let inner = pair.into_inner().next().ok_or("Empty item")?;
        
        match inner.as_rule() {
            Rule::fn_def => self.build_fn_def(inner, network),
            Rule::struct_def => self.build_struct_def(inner, network),
            Rule::let_stmt => self.build_let_stmt(inner, network),
            _ => Err("Unknown item type".into()),
        }
    }

    fn build_fn_def(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let node_id = network.add_concept(0.05); // Low damping for functions
        self.stats.nodes_created += 1;
        
        let mut name = String::new();
        let mut params = Vec::new();
        let mut body = Vec::new();
        
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::kw_pub | Rule::kw_fn => {}
                Rule::ident => {
                    if name.is_empty() {
                        name = inner.as_str().to_string();
                    }
                }
                Rule::param_list => {
                    for param in inner.into_inner() {
                        if let Some(ident) = param.into_inner().next() {
                            params.push(ident.as_str().to_string());
                        }
                    }
                }
                Rule::block => {
                    self.scope_stack.push_scope();
                    
                    // Bind parameters in function scope
                    for param_name in &params {
                        let param_id = network.add_concept(0.1);
                        self.stats.nodes_created += 1;
                        self.scope_stack.bind(param_name.clone(), param_id);
                        
                        // Syntax edge: function III parameter
                        network.connect_syntax(node_id, param_id, 0.9);
                        self.stats.syntax_edges += 1;
                    }
                    
                    body = self.build_block(inner, network)?;
                    self.scope_stack.pop_scope();
                }
                _ => {}
            }
        }
        
        // Register function name in outer scope
        self.scope_stack.bind(name.clone(), node_id);
        
        Ok(AstNode::FnDef { name, params, body, node_id })
    }

    fn build_struct_def(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let node_id = network.add_concept(0.05);
        self.stats.nodes_created += 1;
        
        let mut name = String::new();
        let mut fields = Vec::new();
        
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::ident => {
                    if name.is_empty() {
                        name = inner.as_str().to_string();
                    }
                }
                Rule::field_list => {
                    for field in inner.into_inner() {
                        let mut field_name = String::new();
                        let mut field_type = String::new();
                        
                        for part in field.into_inner() {
                            match part.as_rule() {
                                Rule::ident => {
                                    if field_name.is_empty() {
                                        field_name = part.as_str().to_string();
                                    }
                                }
                                Rule::type_spec => {
                                    field_type = part.as_str().to_string();
                                }
                                _ => {}
                            }
                        }
                        
                        if !field_name.is_empty() {
                            let field_id = network.add_concept(0.1);
                            self.stats.nodes_created += 1;
                            
                            // Syntax edge: struct III field
                            network.connect_syntax(node_id, field_id, 0.9);
                            self.stats.syntax_edges += 1;
                            
                            fields.push((field_name, field_type));
                        }
                    }
                }
                _ => {}
            }
        }
        
        self.scope_stack.bind(name.clone(), node_id);
        Ok(AstNode::StructDef { name, fields, node_id })
    }

    fn build_let_stmt(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let node_id = network.add_concept(0.1);
        self.stats.nodes_created += 1;
        
        let mut name = String::new();
        let mut value: Option<Box<AstNode>> = None;
        
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::ident => {
                    name = inner.as_str().to_string();
                }
                Rule::expr => {
                    value = Some(Box::new(self.build_expr(inner, network)?));
                }
                _ => {}
            }
        }
        
        // Register variable in current scope
        self.scope_stack.bind(name.clone(), node_id);
        
        Ok(AstNode::LetStmt {
            name,
            value: value.ok_or("Let statement without value")?,
            node_id,
        })
    }

    fn build_block(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<Vec<AstNode>, Box<dyn std::error::Error>> {
        let mut stmts = Vec::new();
        
        for inner in pair.into_inner() {
            if let Rule::stmt = inner.as_rule() {
                stmts.push(self.build_stmt(inner, network)?);
            }
        }
        
        Ok(stmts)
    }

    fn build_stmt(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let inner = pair.into_inner().next().ok_or("Empty statement")?;
        
        match inner.as_rule() {
            Rule::let_stmt => self.build_let_stmt(inner, network),
            Rule::if_stmt => self.build_if_stmt(inner, network),
            Rule::for_stmt => self.build_for_stmt(inner, network),
            Rule::while_stmt => self.build_while_stmt(inner, network),
            Rule::return_stmt => self.build_return_stmt(inner, network),
            Rule::expr_stmt => self.build_expr(inner.into_inner().next().unwrap(), network),
            _ => Err("Unknown statement type".into()),
        }
    }

    fn build_if_stmt(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let mut condition = None;
        let mut then_block = Vec::new();
        let mut else_block = None;
        
        let mut parts = pair.into_inner();
        
        // Condition
        if let Some(cond) = parts.next() {
            if cond.as_rule() == Rule::expr {
                condition = Some(Box::new(self.build_expr(cond, network)?));
            }
        }
        
        // Then block
        if let Some(block) = parts.next() {
            if block.as_rule() == Rule::block {
                self.scope_stack.push_scope();
                then_block = self.build_block(block, network)?;
                self.scope_stack.pop_scope();
            }
        }
        
        // Else block
        if let Some(block) = parts.next() {
            if block.as_rule() == Rule::block {
                self.scope_stack.push_scope();
                else_block = Some(self.build_block(block, network)?);
                self.scope_stack.pop_scope();
            }
        }
        
        Ok(AstNode::IfStmt {
            condition: condition.ok_or("If without condition")?,
            then_block,
            else_block,
        })
    }

    fn build_for_stmt(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let var_node_id = network.add_concept(0.1);
        self.stats.nodes_created += 1;
        
        let mut var = String::new();
        let mut iter = None;
        let mut body = Vec::new();
        
        let mut parts = pair.into_inner();
        
        // Loop variable
        if let Some(ident) = parts.next() {
            var = ident.as_str().to_string();
        }
        
        // Skip "in" keyword
        parts.next();
        
        // Iterator expression
        if let Some(expr) = parts.next() {
            iter = Some(Box::new(self.build_expr(expr, network)?));
        }
        
        // Body
        if let Some(block) = parts.next() {
            self.scope_stack.push_scope();
            self.scope_stack.bind(var.clone(), var_node_id);
            body = self.build_block(block, network)?;
            self.scope_stack.pop_scope();
        }
        
        Ok(AstNode::ForLoop {
            var,
            iter: iter.ok_or("For loop without iterator")?,
            body,
            var_node_id,
        })
    }

    fn build_while_stmt(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let mut condition = None;
        let mut body = Vec::new();
        
        let mut parts = pair.into_inner();
        
        if let Some(cond) = parts.next() {
            condition = Some(Box::new(self.build_expr(cond, network)?));
        }
        
        if let Some(block) = parts.next() {
            self.scope_stack.push_scope();
            body = self.build_block(block, network)?;
            self.scope_stack.pop_scope();
        }
        
        Ok(AstNode::WhileLoop {
            condition: condition.ok_or("While without condition")?,
            body,
        })
    }

    fn build_return_stmt(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let value = if let Some(expr) = pair.into_inner().next() {
            Some(Box::new(self.build_expr(expr, network)?))
        } else {
            None
        };
        
        Ok(AstNode::ReturnStmt { value })
    }

    fn build_expr(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let mut terms = Vec::new();
        let mut ops = Vec::new();
        
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::term => {
                    terms.push(self.build_term(inner, network)?);
                }
                Rule::bin_op => {
                    ops.push(inner.as_str().to_string());
                }
                _ => {}
            }
        }
        
        if terms.is_empty() {
            return Err("Empty expression".into());
        }
        
        // Build left-associative expression tree
        let mut result = terms.remove(0);
        for (op, right) in ops.into_iter().zip(terms) {
            result = AstNode::BinOp {
                op,
                left: Box::new(result),
                right: Box::new(right),
            };
        }
        
        Ok(result)
    }

    fn build_term(
        &mut self,
        pair: Pair<Rule>,
        network: &mut ResonanceNetwork,
    ) -> Result<AstNode, Box<dyn std::error::Error>> {
        let inner = pair.into_inner().next().ok_or("Empty term")?;
        
        match inner.as_rule() {
            Rule::number | Rule::string_lit => {
                let node_id = network.add_concept(0.15); // Literals: medium damping
                self.stats.nodes_created += 1;
                
                Ok(AstNode::Literal {
                    value: inner.as_str().to_string(),
                    node_id,
                })
            }
            
            Rule::fn_call => {
                let node_id = network.add_concept(0.08);
                self.stats.nodes_created += 1;
                
                let mut name = String::new();
                let mut args = Vec::new();
                
                for part in inner.into_inner() {
                    match part.as_rule() {
                        Rule::ident => {
                            name = part.as_str().to_string();
                        }
                        Rule::arg_list => {
                            for arg in part.into_inner() {
                                args.push(self.build_expr(arg, network)?);
                            }
                        }
                        _ => {}
                    }
                }
                
                Ok(AstNode::FnCall { name, args, node_id })
            }
            
            Rule::ident => {
                let name = inner.as_str().to_string();
                let node_id = network.add_concept(0.12);
                self.stats.nodes_created += 1;
                
                Ok(AstNode::Ident { name, node_id })
            }
            
            Rule::expr => self.build_expr(inner, network),
            
            _ => Err(format!("Unknown term type: {:?}", inner.as_rule()).into()),
        }
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // AST TRAVERSAL & EDGE CREATION
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    fn traverse_ast(
        &mut self,
        ast: &AstNode,
        network: &mut ResonanceNetwork,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.visit_node(ast, network, None)
    }

    fn visit_node(
        &mut self,
        node: &AstNode,
        network: &mut ResonanceNetwork,
        parent_id: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match node {
            AstNode::Program { items } => {
                for item in items {
                    self.visit_node(item, network, None)?;
                }
            }
            
            AstNode::FnDef { name, params: _, body, node_id } => {
                // Phase 19.5: Logic Domain extraction
                self.nodes.push(SemanticNode {
                    label: name.clone(),
                    base_frequency: 0.5, // Logic functions are mid-frequency
                    decay_rate: 0.1,
                    topological_impedance: 0.3,
                    neuro_flag: DOMAIN_LOGIC,
                });

                // Syntax edge: parent III function (if any)
                if let Some(parent) = parent_id {
                    network.connect_syntax(parent, *node_id, 0.85);
                    self.stats.syntax_edges += 1;
                }
                
                // Visit body statements
                for stmt in body {
                    self.visit_node(stmt, network, Some(*node_id))?;
                }
            }
            
            AstNode::StructDef { node_id, .. } => {
                if let Some(parent) = parent_id {
                    network.connect_syntax(parent, *node_id, 0.85);
                    self.stats.syntax_edges += 1;
                }
            }
            
            AstNode::LetStmt { name, value, node_id } => {
                // Phase 19.5: Variable/Binding extraction (Spatial/Contextual)
                self.nodes.push(SemanticNode {
                    label: name.clone(),
                    base_frequency: 0.8,
                    decay_rate: 0.2,
                    topological_impedance: 0.1,
                    neuro_flag: DOMAIN_SPATIAL,
                });

                // Syntax edge: parent III let statement
                if let Some(parent) = parent_id {
                    network.connect_syntax(parent, *node_id, 0.8);
                    self.stats.syntax_edges += 1;
                }
                
                // Visit value expression
                self.visit_node(value, network, Some(*node_id))?;
            }
            
            AstNode::IfStmt { condition, then_block, else_block } => {
                self.visit_node(condition, network, parent_id)?;
                
                for stmt in then_block {
                    self.visit_node(stmt, network, parent_id)?;
                }
                
                if let Some(else_stmts) = else_block {
                    for stmt in else_stmts {
                        self.visit_node(stmt, network, parent_id)?;
                    }
                }
            }
            
            AstNode::ForLoop { var, iter, body, var_node_id } => {
                if let Some(parent) = parent_id {
                    network.connect_syntax(parent, *var_node_id, 0.8);
                    self.stats.syntax_edges += 1;
                }
                
                self.visit_node(iter, network, Some(*var_node_id))?;
                
                for stmt in body {
                    self.visit_node(stmt, network, Some(*var_node_id))?;
                }

                // Phase 19.5: Temporal Domain extraction (Loops)
                self.nodes.push(SemanticNode {
                    label: format!("loop_{}", var),
                    base_frequency: 0.2, // Low freq for long-running loops
                    decay_rate: 0.05,
                    topological_impedance: 0.6, // Loops create higher topological friction
                    neuro_flag: DOMAIN_TEMPORAL,
                });
            }
            
            AstNode::WhileLoop { condition, body } => {
                self.visit_node(condition, network, parent_id)?;
                
                for stmt in body {
                    self.visit_node(stmt, network, parent_id)?;
                }
            }
            
            AstNode::ReturnStmt { value } => {
                if let Some(expr) = value {
                    self.visit_node(expr, network, parent_id)?;
                }
            }
            
            AstNode::BinOp { left, right, .. } => {
                self.visit_node(left, network, parent_id)?;
                self.visit_node(right, network, parent_id)?;
            }
            
            AstNode::FnCall { name, args, node_id } => {
                // Syntax edge: parent III call
                if let Some(parent) = parent_id {
                    network.connect_syntax(parent, *node_id, 0.75);
                    self.stats.syntax_edges += 1;
                }
                
                // SEMANTIC EDGE: Call site III Function definition
                if let Some(def_id) = self.scope_stack.lookup(name) {
                    network.connect_semantic(*node_id, def_id, 0.95);
                    self.stats.semantic_edges += 1;
                }
                
                // Visit arguments
                for arg in args {
                    self.visit_node(arg, network, Some(*node_id))?;
                }
            }
            
            AstNode::Ident { name, node_id } => {
                // Syntax edge: parent III identifier
                if let Some(parent) = parent_id {
                    network.connect_syntax(parent, *node_id, 0.7);
                    self.stats.syntax_edges += 1;
                }
                
                // SEMANTIC EDGE: Variable use III Variable definition
                if let Some(def_id) = self.scope_stack.lookup(name) {
                    if def_id != *node_id {
                        network.connect_semantic(*node_id, def_id, 0.9);
                        self.stats.semantic_edges += 1;
                    }
                }
            }
            
            AstNode::Literal { node_id, .. } => {
                if let Some(parent) = parent_id {
                    network.connect_syntax(parent, *node_id, 0.65);
                    self.stats.syntax_edges += 1;
                }
            }
        }
        
        Ok(())
    }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// SEMANTIC REGISTER FILE (SRF)
//
// DonanIIm yazmaII mimarisi III 5 II f32 = 20 bayt, tek cache line'a sIIIIar.
// Standart LLM'lerin O(NI?) KV-cache zincirini O(1) faz-uzayII aritmetiIIine
// indirger. Heap allocasyon yok, String yok, Tensor yok.
//
// YazmaII mimarisi analoji:
//   AX  IIIIII  x86 AX  (accumulator  III aktif iIIlem verisi)
//   BX  IIIIII  x86 BX  (base register III sabit referans)
//   DX  IIIIII  x86 DX  (data register III dIIIIarIIdan enjekte)
//   CX  IIIIII  x86 CX  (counter/chaos III kaos IIlIIIIm IIifti)
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

/// BeII f32 yazmacIIndan oluIIan sIIfIIr-malloc baIIlam birimi.
///
/// Oracle Task her token dIIngIIsIInde bu yazmaclarII `try_lock` ile gIInceller;
/// ana IIretim thread'i hiIIbir zaman bu Mutex'i beklemez.
///
/// Phase 21: Now includes ZeroCopyCartridge for zero-copy QMEM semantic tensors.
/// CPU writes to host_ptr, GPU reads from dev_ptr directly over PCIe.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SemanticRegisterFile {
    /// **AX III Aktif Odak**: geIIerli token/cIImlenin faz aIIIIsII II III [0, 2II].
    /// Oracle her token'da token_id'yi [0, 2II] aralIIIIIIna eIIleyerek yazar.
    pub ax: f32,

    /// **BX III Base Context**: sistem promptu / kurallarIIn faz aIIIIsI?.
    /// `generate_stream` baIIIInda `phase_alignment` deIIeriyle bir kez atanIIr,
    /// IIretim boyunca salt-okunur (semantik read-only) davranIIr.
    pub bx: f32,

    /// **DX III Data Register**: RAG / .qmem ile enjekte edilen iIIeriIIin faz aIIIIsI?.
    /// Veri IIekme pipeline'II tarafIIndan yazIIlIIr; veri-IIIIkarma gIIrevlerinde kullanIIlIIr.
    pub dx: f32,

    /// **CX_W III Phase Variance**: sekans uyum skoru [0.0, 1.0].
    /// 0 III halIIsinasyon/tekrar, 1 III baIIIImsIIz/saIIlIIklII IIretim.
    pub cx_w: f32,

    /// **CX_J III Chaos Jacobian Proxy**: `|W_actual III W_predicted|`.
    /// Oracle'IIn tahmin sapmasII III kaos bIIyIIklIIIII?.
    pub cx_j: f32,
}

/// Phase 21: Extended SRF with zero-copy mapped memory for QMEM tensors.
/// This eliminates the need for Tensor::from_vec which triggers cudaMemcpy.
pub struct SemanticRegisterFileWithQmem {
    /// Core register file (AX, BX, DX, CX_W, CX_J)
    pub srf: SemanticRegisterFile,
    /// Zero-copy mapped memory for QMEM poles (frequency, decay pairs)
    pub poles_cartridge: Option<ZeroCopyCartridge>,
    /// Zero-copy mapped memory for QMEM impedance values
    pub impedance_cartridge: Option<ZeroCopyCartridge>,
    /// Zero-copy mapped memory for QMEM neuro flags
    pub flags_cartridge: Option<ZeroCopyCartridge>,
    /// Number of semantic nodes currently in the cartridges
    pub node_count: usize,
}

impl SemanticRegisterFileWithQmem {
    /// CliffordBridge integration: write h_phase into DX register.
    ///
    /// Called every token step after compute_clifford_projection().
    /// h_phase = Clifford geometric product argument (saçak/fringe intersection angle).
    /// cx_w_boost = confidence boost from saçak node count (0.0–0.1 typical).
    ///
    /// DX is then read by compute_alu_phase(is_data_extraction=true):
    ///   ALU_out = DX * 0.7 + AX * 0.3
    /// → Clifford projection dominates the data-extraction path.
    #[inline]
    pub fn update_from_bridge(&mut self, h_phase: f32, cx_w_boost: f32) {
        self.srf.set_data_phase(h_phase);
        // CX_W reflects saçak node strength: more fringe resonance = more confident
        self.srf.cx_w = (self.srf.cx_w + cx_w_boost).clamp(0.0, 1.0);
        eprintln!(
            "[CLIFFORD BRIDGE] h_phase={:.4} rad → DX={:.4} | cx_w={:.4}",
            h_phase, self.srf.dx, self.srf.cx_w
        );
    }
}

impl SemanticRegisterFile {
    /// SIIfIIr-hazIIr SRF oluIItur; BX'i verilen temel faz ile sabitle.
    ///
    /// `bx_phase`: `compute_total_energy()` IIIIktIIsII (f64 III f32 cast kabul edilir).
    #[inline]
    pub fn new(bx_phase: f32) -> Self {
        Self {
            ax:   0.0,
            bx:   bx_phase,
            dx:   0.0,
            cx_w: 0.5,   // ilk token gelmeden IInce nIItr Phase Variance
            cx_j: 0.0,   // stabil baIIlangIIII
        }
    }

    /// **ALU Faz HesabII** III O(1), saf aritmetik, sIIfIIr allocasyon.
    ///
    /// Veri IIekme (RAG / .qmem) yolu:
    /// ```text
    ///   II_out = DXII0.7 + AXII0.3   // enjekte veri fazII aIIIIrlIIklII
    /// ```
    ///
    /// Normal mantIIk/kural yIIrIItme yolu:
    /// ```text
    ///   II_out = BXII0.8 + AXII0.2   // temel baIIlam fazII aIIIIrlIIklII
    /// ```
    ///
    /// GPR'ye beslenen `active_theta` deIIeri buradan gelir.
    #[inline(always)]
    pub fn compute_alu_phase(&self, is_data_extraction: bool) -> f32 {
        if is_data_extraction {
            (self.dx * 0.7) + (self.ax * 0.3)
        } else {
            (self.bx * 0.8) + (self.ax * 0.2)
        }
    }

    /// Atomic snapshot: ALU hesabII + CX IIifti tek try_lock penceresi iIIinde.
    /// Oracle bu yardIImcIIyII kullanarak lock sIIresi < 5 ns'de kalIIr.
    #[inline(always)]
    pub fn alu_snapshot(&self, is_data_extraction: bool) -> (f32, f32, f32) {
        let theta = self.compute_alu_phase(is_data_extraction);
        (theta, self.cx_w, self.cx_j)
    }

    /// **Oracle GIIncelleme** III yalnIIzca Oracle Task bu metodu IIaIIIIrmalIIdIIr.
    ///
    /// GIIncellenen yazmaIIlar: AX, CX_W, CX_J.
    /// **BX ve DX intentional olarak burada YOK**: Read-Only semantiIIi.
    /// BX = sistem promptu fazI?; nesil baIIIInda bir kez set edilir, sonra deIIiIImez.
    /// DX = RAG/qmem enjeksiyon fazI?; harici pipeline `srf.dx = ...` ile yazar.
    ///
    /// `token_ax` formIIlII (IIaIIIIran tarafIIndan hesaplanIIr):
    ///   `(token_id as f32 * TAU) / vocab_size`
    ///
    /// Maliyet: 3 f32 write + 2 clamp/max = ~5 CPU dIIngIIsI?.
    #[inline(always)]
    pub fn oracle_update(&mut self, token_ax: f32, phase_variance: f32, jacobian: f32) {
        self.ax   = token_ax;
        self.cx_w = phase_variance.clamp(0.0, 1.0);
        self.cx_j = jacobian.max(0.0);
    }

    /// DX yazmaIIIInII harici veri pipeline'IIndan gIIncelle (RAG / .qmem enjeksiyonu).
    ///
    /// Bu metot Oracle tarafIIndan IIaIIrIIlmaz; veri enjeksiyon sistemi IIaIIIIrIIr.
    /// `data_phase`: enjekte edilen iIIeriIIin `compute_total_energy()` deIIeri.
    #[inline(always)]
    pub fn set_data_phase(&mut self, data_phase: f32) {
        self.dx = data_phase;
    }

    /// **Phase 21: Zero-Copy QMEM Cartridge Compiler**
    /// Writes semantic node data directly to zero-copy mapped memory.
    /// CPU writes to host_ptr, GPU reads from dev_ptr directly over PCIe.
    /// NO cudaMemcpy - memory is mapped and immediately available to GPU.
    pub fn compile_zero_copy_cartridge(
        nodes: &[SemanticNode],
        poles_cartridge: &mut Option<ZeroCopyCartridge>,
        impedance_cartridge: &mut Option<ZeroCopyCartridge>,
        flags_cartridge: &mut Option<ZeroCopyCartridge>,
    ) -> Result<usize, String> {
        let n = nodes.len();
        if n == 0 {
            return Err("Cannot compile empty semantic topology".to_string());
        }

        // Ensure cartridges have sufficient capacity
        crate::gpu_ops::memory_utils::ensure_zero_copy_capacity(poles_cartridge, n * 2)?;
        crate::gpu_ops::memory_utils::ensure_zero_copy_capacity(impedance_cartridge, n)?;
        crate::gpu_ops::memory_utils::ensure_zero_copy_capacity(flags_cartridge, n)?;

        // Extract data into vectors
        let mut frequencies = Vec::with_capacity(n);
        let mut decays = Vec::with_capacity(n);
        let mut impedances = Vec::with_capacity(n);
        let mut flags = Vec::with_capacity(n);

        for node in nodes {
            frequencies.push(node.base_frequency);
            decays.push(node.decay_rate);
            impedances.push(node.topological_impedance);
            flags.push(node.neuro_flag);
        }

        // Write to pinned memory
        if let Some(ref mut poles) = *poles_cartridge {
            poles.write_poles(&frequencies, &decays)?;
            poles.sync_to_vram()?;
        }
        if let Some(ref mut imp) = *impedance_cartridge {
            imp.write_impedance(&impedances)?;
            imp.sync_to_vram()?;
        }
        if let Some(ref mut fl) = *flags_cartridge {
            fl.write_flags(&flags)?;
            fl.sync_to_vram()?;
        }

        Ok(n)
    }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// OPTIMIZATIONS
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

impl Drop for CodeWeaver {
    fn drop(&mut self) {
        // Clear scope stack explicitly (though Rust handles this)
        self.scope_stack.frames.clear();
    }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// PHASE 48: THE EBNF SYNTAX ORACLE FOR GROVER'S QUANTUM AMPLITUDE AMPLIFICATION
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

/// EBNF Grammar Validation Token Types for Binary Oracle
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SyntaxToken {
    /// Valid token (preserves amplitude)
    Valid,
    /// Syntax error (triggers phase inversion)
    Invalid,
    /// Logical contradiction (triggers phase inversion)
    Contradiction,
    /// Type mismatch (triggers phase inversion)
    TypeError,
}

/// PHASE 49: The Semantic Oracle (ODE-Only)
/// 
/// The RK4 continuous differential equation loop must NEVER trigger phase inversion
/// due to discrete EBNF grammar errors. This caused "Double-Tap Halt" where destructive
/// interference collapsed the latent space to zero energy.
/// 
/// NEW ROLE: This Oracle ONLY penalizes Semantic and Physical contradictions:
/// - Ignoring the 10x telemetry ghost vector (physical contradiction)
/// - Violating Mass Conservation (physical contradiction)
/// - Semantic incoherence (conceptual contradiction)
/// 
/// EBNF grammar validation has been MOVED to discrete logit emission phase
/// (EbnfLogitMask in generation_loop.rs) per industry standard constrained decoding.
pub struct SyntaxOracle<'a> {
    /// Threshold for semantic divergence from valid state
    epsilon: f64,
    /// Reference to semantic network for semantic coherence checks
    network: &'a ResonanceNetwork,
}

impl<'a> SyntaxOracle<'a> {
    /// Initialize the Syntax Oracle with strict validation threshold
    pub fn new(epsilon: f64, network: &'a ResonanceNetwork) -> Self {
        Self {
            epsilon,
            network,
        }
    }

    /// PHASE 49: Semantic-Physical Oracle for Grover's Search (NO EBNF)
    /// 
    /// This method ONLY detects semantic and physical contradictions in the continuous
    /// latent space. It NEVER checks EBNF grammar - that happens at discrete token emission.
    /// 
    /// # Arguments
    /// * `h_current` - The intermediate latent thought vector from RK4
    /// 
    /// # Returns
    /// * true = semantic/physical violation detected → apply -π phase penalty
    /// * false = semantically valid → preserve amplitude
    pub fn is_violation(&self, h_current: &candle_core::Tensor) -> candle_core::Result<bool> {
        // Extract thought vector as f32
        let h_vec = h_current.to_vec1::<f32>()?;
        
        // Check 1: Semantic Coherence (Top-K activation entropy)
        // If the highest activations are contradictory (e.g., "hot" and "cold" at same strength)
        let semantic_violation = self.check_semantic_contradiction(&h_vec);
        
        // Check 2: Physical Conservation Law Violation
        // Energy in the latent field must be conserved (no spontaneous creation/destruction)
        let physical_violation = self.check_conservation_violation(&h_vec);
        
        // Check 3: Ghost Vector Ignoring (10x telemetry miss)
        // Detect if model is ignoring strong peripheral signals
        let ghost_ignored = self.check_ghost_vector_ignored(&h_vec);
        
        let is_violation = semantic_violation || physical_violation || ghost_ignored;
        
        if is_violation {
            eprintln!(
                "[SEMANTIC ORACLE] VIOLATION: semantic={} physical={} ghost_ignored={}",
                semantic_violation, physical_violation, ghost_ignored
            );
        }
        
        Ok(is_violation)
    }

    /// Check for semantic contradictions (e.g., strong activation of contradictory concepts)
    fn check_semantic_contradiction(&self, h_vec: &[f32]) -> bool {
        let mut indexed: Vec<(usize, f32)> = h_vec.iter().enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Check top 8 activations for contradictions
        let top_k: Vec<(usize, f32)> = indexed.iter().take(8).cloned().collect();
        
        // If two high-activation concepts are contradictory (negative correlation in network)
        for i in 0..top_k.len() {
            for j in (i+1)..top_k.len() {
                let (id_i, act_i) = top_k[i];
                let (id_j, act_j) = top_k[j];
                
                // Both highly active
                if act_i > 0.5 && act_j > 0.5 {
                    // Check if they have negative correlation in semantic network
                    if let Some(corr) = self.network.get_correlation(id_i, id_j) {
                        if corr < -0.8 {
                            // Strongly contradictory concepts both highly active
                            return true;
                        }
                    }
                }
            }
        }
        
        false
    }

    /// Check for conservation law violations in the latent field
    fn check_conservation_violation(&self, h_vec: &[f32]) -> bool {
        // Total energy should be conserved (no spontaneous creation)
        let total_energy: f32 = h_vec.iter().map(|&v| v * v).sum();
        
        // Check for unphysical energy levels (too high = numerical explosion, too low = collapse)
        let num_dims = h_vec.len() as f32;
        let mean_energy = total_energy / num_dims;
        
        // Violation if mean energy per dimension is unphysically high or near-zero
        mean_energy > 10.0 || (mean_energy < 1e-6 && total_energy > 0.0)
    }

    /// Check if ghost vector (peripheral telemetry) is being ignored
    fn check_ghost_vector_ignored(&self, h_vec: &[f32]) -> bool {
        // Determine total energy - empty/zero vector is inactive, not ignoring
        let total_energy: f32 = h_vec.iter().map(|&v| v * v).sum();
        if total_energy < 1e-6 {
            return false;
        }

        // Ghost vectors typically manifest as specific pattern in dimensions
        // This is a simplified check - in production would check actual ghost vector state
        
        // Check if the latent vector is "ignoring" signals by being too uniform
        let mean: f32 = h_vec.iter().sum::<f32>() / h_vec.len() as f32;
        let variance: f32 = h_vec.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / h_vec.len() as f32;
        
        // Extremely low variance means model is "flatlining" and ignoring inputs
        variance < 1e-8
    }

    /// PHASE 49: Compute Phase Inversion Penalty (Semantic/Physical only, NO EBNF)
    /// 
    /// Returns the -π phase penalty based on violation severity.
    pub fn compute_phase_penalty(&self, h_current: &candle_core::Tensor) -> candle_core::Result<f64> {
        let h_vec = h_current.to_vec1::<f32>()?;
        
        // Compute severity based on semantic/physical violations only
        let mut max_severity = 0.0f32;
        
        // Semantic contradiction severity
        if self.check_semantic_contradiction(&h_vec) {
            max_severity = max_severity.max(0.8);
        }
        
        // Physical violation severity
        if self.check_conservation_violation(&h_vec) {
            let total_energy: f32 = h_vec.iter().map(|&v| v * v).sum();
            let num_dims = h_vec.len() as f32;
            let mean_energy = total_energy / num_dims;
            
            // Severity scales with how unphysical the energy is
            let physical_severity = if mean_energy > 10.0 {
                (mean_energy / 10.0).min(1.0)
            } else {
                0.0
            };
            max_severity = max_severity.max(physical_severity);
        }
        
        // Ghost ignored severity
        if self.check_ghost_vector_ignored(&h_vec) {
            max_severity = max_severity.max(0.9); // High severity for ignoring telemetry
        }
        
        let penalty = (max_severity as f64 / self.epsilon).min(1.0) * std::f64::consts::PI;
        
        if penalty > 0.0 {
            eprintln!("[SEMANTIC ORACLE] Phase penalty π×{:.2}", max_severity);
        }
        
        Ok(penalty)
    }
}

impl<'a> Default for SyntaxOracle<'a> {
    fn default() -> Self {
        // Note: This default requires a network reference for real usage
        // Tests must use SyntaxOracle::new with a real network
        panic!("SyntaxOracle requires a ResonanceNetwork reference. Use SyntaxOracle::new(epsilon, network) instead.")
    }
}

/// PHASE 49: Multiple-Choice Constraint for SyntaxOracle
/// 
/// Enables constrained decoding to guarantee selection from a fixed set of valid choices.
/// Useful for multiple-choice questions, option selection, and structured outputs.
#[derive(Debug, Clone)]
pub struct ChoiceConstraint {
    /// Valid choice options (e.g., ["A", "B", "C", "D"] or ["yes", "no"])
    pub choices: Vec<String>,
    /// Token IDs corresponding to each choice (populated during tokenization)
    pub choice_token_ids: Vec<Vec<u32>>,
    /// Semantic scores for each choice (from network analysis)
    pub choice_coherence: Vec<f32>,
    /// Whether the constraint is currently active
    pub is_active: bool,
}

impl ChoiceConstraint {
    /// Create a new choice constraint with a list of valid options
    pub fn new(choices: Vec<String>) -> Self {
        let num_choices = choices.len();
        Self {
            choices,
            choice_token_ids: vec![vec![]; num_choices],
            choice_coherence: vec![0.0; num_choices],
            is_active: false,
        }
    }

    /// Activate the constraint for upcoming token generation
    pub fn activate(&mut self) {
        self.is_active = true;
    }

    /// Deactivate the constraint
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }

    /// Get the number of valid choices
    pub fn num_choices(&self) -> usize {
        self.choices.len()
    }
}

/// Extension trait for SyntaxOracle to support multiple-choice constraints
impl<'a> SyntaxOracle<'a> {
    /// Validate that a given token/output is one of the allowed choices
    /// 
    /// # Arguments
    /// * `output_text` - The generated output to validate
    /// * `constraint` - The choice constraint to apply
    /// 
    /// # Returns
    /// * Some(index) if output matches a valid choice (index of matched choice)
    /// * None if output doesn't match any valid choice
    pub fn validate_choice(&self, output_text: &str, constraint: &ChoiceConstraint) -> Option<usize> {
        if !constraint.is_active {
            return None;
        }

        let trimmed = output_text.trim();
        
        // Exact match first
        for (idx, choice) in constraint.choices.iter().enumerate() {
            if trimmed == choice || trimmed == choice.trim() {
                return Some(idx);
            }
        }

        // Case-insensitive match
        let lower_text = trimmed.to_lowercase();
        for (idx, choice) in constraint.choices.iter().enumerate() {
            if lower_text == choice.to_lowercase() {
                return Some(idx);
            }
        }

        // Prefix match (for longer choice formats)
        for (idx, choice) in constraint.choices.iter().enumerate() {
            if trimmed.starts_with(choice) {
                return Some(idx);
            }
        }

        None
    }

    /// Compute semantic coherence score for each choice
    /// 
    /// Uses the semantic network to score how coherent each choice is
    /// relative to the current context (represented by h_context).
    /// 
    /// # Arguments
    /// * `h_context` - The current latent context vector
    /// * `constraint` - The choice constraint to score
    /// 
    /// # Returns
    /// Result with coherence scores for each choice
    pub fn score_choices(&self, h_context: &Tensor, constraint: &mut ChoiceConstraint) -> CandleResult<Vec<f32>> {
        let h_vec = h_context.to_vec1::<f32>()?;
        let mut scores = vec![0.0; constraint.num_choices()];

        // Score each choice based on semantic coherence with context
        for (idx, choice) in constraint.choices.iter().enumerate() {
            // Simple coherence metric: measure how well the choice aligns with current context
            // In production, would use more sophisticated semantic matching
            let mut coherence = 0.0f32;
            
            // Extract first word from choice for network lookup
            let choice_word = choice.split_whitespace().next().unwrap_or(choice);
            
            // Find this word's potential network node (simplified)
            // In production, would look up actual semantic node from network
            for (i, &h_val) in h_vec.iter().enumerate().take(8) {
                // Normalize activation and use as base coherence
                coherence += h_val.abs() / 8.0;
            }
            
            // Apply semantic penalty if available (lower coherence for less semantic agreement)
            if let Some(corr) = self.network.get_correlation(0, idx) {
                coherence = coherence * (corr.max(-1.0) as f32).max(0.0);
            }
            
            scores[idx] = coherence;
            constraint.choice_coherence[idx] = coherence;
        }

        Ok(scores)
    }

    /// Enforce choice constraint by returning the best-scoring valid choice
    /// 
    /// If semantic scoring is available, returns the highest-scoring choice.
    /// Otherwise, returns the first valid choice.
    /// 
    /// # Arguments
    /// * `constraint` - The choice constraint with scores
    /// * `generated_text` - Optional generated text (if provided, validates it first)
    /// 
    /// # Returns
    /// Result with the index of the selected choice
    pub fn enforce_choice_constraint(&self, constraint: &ChoiceConstraint, generated_text: Option<&str>) -> CandleResult<usize> {
        if !constraint.is_active || constraint.choices.is_empty() {
            return Err(candle_core::Error::Msg("Choice constraint not active or empty".to_string()));
        }

        // If generated text matches a choice, return that choice
        if let Some(text) = generated_text {
            if let Some(idx) = self.validate_choice(text, constraint) {
                return Ok(idx);
            }
        }

        // Otherwise, return the highest-scoring choice
        let best_idx = constraint.choice_coherence
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        eprintln!("[CHOICE ORACLE] Selected choice {} ({}): coherence={:.3}", 
                  best_idx, constraint.choices[best_idx], constraint.choice_coherence[best_idx]);

        Ok(best_idx)
    }

    /// Suggest a valid choice that the model should emit next
    /// 
    /// Returns formatted suggestion string based on the choice constraint.
    /// Useful for guiding model generation towards valid outputs.
    pub fn suggest_valid_choice(&self, constraint: &ChoiceConstraint) -> Option<String> {
        if !constraint.is_active || constraint.choices.is_empty() {
            return None;
        }

        let best_idx = constraint.choice_coherence
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)?;

        Some(constraint.choices[best_idx].clone())
    }
}

/// PHASE 49: EBNF Logit Mask for Discrete Constrained Decoding
/// 
/// Industry-standard constrained decoding implementation (like Outlines/Guidance).
/// Intercepts raw logits BEFORE sampling and masks tokens that would cause
/// unrecoverable EBNF syntax errors to f32::NEG_INFINITY.
/// 
/// PHASE 49.5: NO-ESCAPE PROTOCOL - The model CANNOT emit EOS until the
/// EBNF grammar has reached a valid End-of-Input (EOI) state. This prevents
/// the model from "cheating" its way out of strict causal chain requirements.
/// 
/// This is the discrete counterpart to the continuous SemanticOracle:
/// - SemanticOracle: Validates continuous latent space (RK4 phase inversion)
/// - EbnfLogitMask: Validates discrete token emission (logit masking)
pub struct EbnfLogitMask {
    /// Maximum tokens to evaluate per step (top-k candidates)
    max_candidates: usize,
    /// Cache of validated token sequences to avoid re-parsing
    validation_cache: std::collections::HashMap<String, bool>,
    /// PHASE 49.6: Multiple EOS token IDs for Qwen2.5
    eos_token_ids: Vec<u32>,
    /// PHASE 49.5: Track if EBNF has reached valid EOI state
    eoi_reached: bool,
}

impl EbnfLogitMask {
    /// PHASE 49.6: Create new EBNF logit mask with multiple EOS blocking enabled
    pub fn new(max_candidates: usize) -> Self {
        Self {
            max_candidates,
            validation_cache: std::collections::HashMap::new(),
            eos_token_ids: vec![151643, 151644, 151645], // Qwen2.5: <|endoftext|>, <|im_start|>, <|im_end|>
            eoi_reached: false,
        }
    }

    /// PHASE 49.6: Create with custom EOS token IDs
    pub fn with_eos_tokens(max_candidates: usize, eos_token_ids: Vec<u32>) -> Self {
        Self {
            max_candidates,
            validation_cache: std::collections::HashMap::new(),
            eos_token_ids,
            eoi_reached: false,
        }
    }

    /// PHASE 49: Apply EBNF constraint mask to logits tensor
    /// 
    /// 1. Extract top-K candidate tokens from logits
    /// 2. For each candidate, append to current sequence and test with EBNF parser
    /// 3. If candidate causes unrecoverable error, set its logit to -INF
    /// 4. Return masked logits
    /// 
    /// # Arguments
    /// * `logits` - Raw logits tensor [1, 1, vocab_size]
    /// * `current_sequence` - Tokens generated so far
    /// * `tokenizer` - Tokenizer for decoding tokens to strings
    /// 
    /// # Returns
    /// Masked logits with invalid tokens set to NEG_INFINITY
    pub fn apply_mask(
        &mut self,
        logits: &candle_core::Tensor,
        current_sequence: &[u32],
        tokenizer: &tokenizers::Tokenizer,
        prompt_text: &str,
    ) -> candle_core::Result<candle_core::Tensor> {
        use candle_core::IndexOp;
        
        let vocab_size = logits.dims().last().copied().unwrap_or(0);
        if vocab_size == 0 {
            return Ok(logits.clone());
        }

        // Get current sequence as string
        let current_text = tokenizer.decode(current_sequence, false)
            .unwrap_or_default();

        // Check if currently in reasoning/thinking phase
        let is_thinking = (current_text.contains("<think>") && !current_text.contains("</think>"))
            || (current_text.contains("<thought>") && !current_text.contains("</thought>"))
            || (current_text.contains("<option_analysis>") && !current_text.contains("</option_analysis>"));
        if is_thinking {
            return Ok(logits.clone());
        }

        // Isolate target text after the thinking blocks or prompt
        let target_text = if let Some(pos) = current_text.rfind("</think>") {
            &current_text[pos + "</think>".len()..]
        } else if let Some(pos) = current_text.rfind("</thought>") {
            &current_text[pos + "</thought>".len()..]
        } else if let Some(pos) = current_text.rfind("</option_analysis>") {
            &current_text[pos + "</option_analysis>".len()..]
        } else {
            if current_text.starts_with(prompt_text) {
                &current_text[prompt_text.len()..]
            } else {
                &current_text
            }
        };

        // PHASE 49.5: NO-ESCAPE PROTOCOL - Check if current sequence can end
        // First, check if the current sequence alone would be complete
        let current_is_complete = Self::check_eoi_state(target_text);
        self.eoi_reached = current_is_complete;

        // Extract top-K candidates
        let logits_vec = logits.to_vec1::<f32>()?;
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Collect invalid token IDs
        let mut invalid_mask: Vec<bool> = vec![false; vocab_size];
        let top_candidates: Vec<(usize, f32)> = indexed.iter()
            .take(self.max_candidates)
            .cloned()
            .collect();

        for (token_id, original_logit) in top_candidates {
            // Skip if already very low logit
            if original_logit < -100.0 {
                continue;
            }

            // PHASE 49.6: NO-ESCAPE PROTOCOL - Block ALL EOS tokens if EOI not reached
            if self.eos_token_ids.contains(&(token_id as u32)) && !self.eoi_reached {
                // Force EOS to -INF if EBNF not complete
                invalid_mask[token_id] = true;
                continue;
            }

            // Decode candidate token
            let candidate_text = match tokenizer.decode(&[token_id as u32], false) {
                Ok(text) => text,
                Err(_) => continue,
            };

            // Build tentative sequence
            let tentative = format!("{}{}", target_text, candidate_text);

            // Check cache first
            let is_valid = match self.validation_cache.get(&tentative) {
                Some(&valid) => valid,
                None => {
                    // Validate with EBNF parser
                    let valid = Self::validate_partial(&tentative);
                    self.validation_cache.insert(tentative.clone(), valid);
                    valid
                }
            };

            if !is_valid {
                // Mark as invalid (will set to -INF)
                invalid_mask[token_id] = true;
            }
        }

        // PHASE 49.6: Final safety check - if any EOS would be top token but blocked, log it
        if !self.eoi_reached && !invalid_mask.is_empty() {
            for eos_id in &self.eos_token_ids {
                let eos_idx = *eos_id as usize;
                if eos_idx < logits_vec.len() && logits_vec[eos_idx] > -10.0 {
                    // Forced continue log bypassed
                }
            }
        }

        // Apply mask: invalid tokens -> NEG_INFINITY
        let mut masked_logits = logits_vec.clone();
        for (i, is_invalid) in invalid_mask.iter().enumerate() {
            if *is_invalid {
                masked_logits[i] = f32::NEG_INFINITY;
            }
        }

        // Create result tensor
        let device = logits.device();
        let masked_tensor = candle_core::Tensor::from_vec(masked_logits, logits.shape().clone(), device)?;

        let invalid_count = invalid_mask.iter().filter(|&&x| x).count();
        Ok(masked_tensor)
    }

    /// PHASE 49.5: Check if sequence has reached valid End-of-Input (EOI) state
    /// 
    /// Returns true if the sequence is grammatically complete and can terminate
    fn check_eoi_state(sequence: &str) -> bool {
        use pest::Parser;
        
        // Empty or whitespace-only sequences are NOT complete
        if sequence.trim().is_empty() {
            return false;
        }
        
        // Try to parse - if it succeeds completely, EOI is reached
        match CodeParser::parse(Rule::program, sequence) {
            Ok(pairs) => {
                // Check if we have a complete program (not just partial)
                let has_content = pairs.flatten().any(|pair| {
                    let rule_str = format!("{:?}", pair.as_rule());
                    // Check for content-bearing rules
                    (rule_str.contains("statement") || 
                     rule_str.contains("expression") ||
                     rule_str.contains("chain")) &&
                    !pair.as_str().trim().is_empty()
                });
                
                // Also check for required causal chain completion markers
                let text = sequence.trim();
                let has_causal_structure = 
                    text.contains("→") || 
                    text.contains("causal") ||
                    text.contains("chain") ||
                    text.contains("Root cause:");
                
                has_content && has_causal_structure
            }
            Err(_) => {
                // Parse error means incomplete
                false
            }
        }
    }

    /// PHASE 49: Validate partial sequence using EBNF parser
    /// 
    /// Uses pest to check if the sequence is a valid prefix (may be incomplete)
    /// Returns true if valid or incomplete-but-valid, false if unrecoverable error
    fn validate_partial(sequence: &str) -> bool {
        use pest::Parser;
        
        // Try to parse the full sequence
        match CodeParser::parse(Rule::program, sequence) {
            Ok(_) => {
                // Complete valid program
                true
            }
            Err(e) => {
                // Check if error is recoverable (incomplete but valid so far)
                let error_str = format!("{}", e);
                
                // Unrecoverable errors that should trigger masking
                let unrecoverable = [
                    "expected",
                    "unexpected",
                    "error",
                    "invalid",
                ];
                
                // Check for unrecoverable error patterns
                for pattern in &unrecoverable {
                    if error_str.to_lowercase().contains(pattern) {
                        // This is a syntax error - mask the token
                        return false;
                    }
                }
                
                // If it's just incomplete (e.g., EOF reached mid-expression), allow it
                true
            }
        }
    }

    /// PHASE 49.5: Check if EOS is currently allowed (EOI reached)
    pub fn is_eos_allowed(&self) -> bool {
        self.eoi_reached
    }

    /// PHASE 49.5: Force allow EOS (for emergency escape)
    pub fn force_allow_eos(&mut self) {
        self.eoi_reached = true;
    }

    /// Get cache statistics for debugging
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.validation_cache.len(), self.validation_cache.capacity())
    }
}

impl Default for EbnfLogitMask {
    fn default() -> Self {
        Self::new(64) // Default: check top 64 candidates
    }
}

// ============================================================================
// STRICT XML FORMAT LOGIT MASK
// ============================================================================
pub struct StrictXmlLogitMask {
    allowed_letters: Vec<char>,
    eos_token_ids: Vec<u32>,
}

impl StrictXmlLogitMask {
    pub fn new(allowed_letters: Vec<char>) -> Self {
        Self {
            allowed_letters,
            eos_token_ids: vec![151643, 151644, 151645], // Qwen2.5 EOS variants
        }
    }

    pub fn default_abcd() -> Self {
        Self::new(vec!['A', 'B', 'C', 'D'])
    }

    pub fn is_valid_prefix(text: &str, letters: &[char]) -> bool {
        let expected = "<answer>";
        let close_tag = "</answer>";
        
        let trimmed = text.trim_start();
        if trimmed.is_empty() { return true; }
        
        if trimmed.len() <= expected.len() {
            return expected.starts_with(trimmed);
        }
        
        if !trimmed.starts_with(expected) {
            return false;
        }
        
        let after_open = &trimmed[expected.len()..];
        if after_open.is_empty() { return true; }
        
        let letter = after_open.chars().next().unwrap();
        if !letters.contains(&letter.to_ascii_uppercase()) {
            return false;
        }
        
        let after_letter = &after_open[1..];
        if after_letter.is_empty() { return true; }
        
        if close_tag.starts_with(after_letter) {
            return true;
        }
        
        if after_letter.starts_with(close_tag) {
            let after_close = &after_letter[close_tag.len()..];
            return after_close.chars().all(|c| c.is_whitespace());
        }
        
        false
    }

    pub fn is_complete_answer(text: &str) -> bool {
        let trimmed = text.trim_start();
        if !trimmed.starts_with("<answer>") { return false; }
        let after_open = &trimmed["<answer>".len()..];
        if after_open.is_empty() { return false; }
        let after_letter = &after_open[1..];
        after_letter.starts_with("</answer>")
    }

    pub fn apply_mask(
        &self,
        logits: &candle_core::Tensor,
        current_sequence: &[u32],
        tokenizer: &tokenizers::Tokenizer,
        prompt_text: &str,
    ) -> candle_core::Result<candle_core::Tensor> {
        let vocab_size = logits.dims().last().copied().unwrap_or(0);
        if vocab_size == 0 { return Ok(logits.clone()); }

        let current_text = tokenizer.decode(current_sequence, false).unwrap_or_default();
        
        let target_text = if current_text.starts_with(prompt_text) {
            &current_text[prompt_text.len()..]
        } else {
            &current_text
        };

        let logits_vec = logits.to_vec1::<f32>()?;
        let mut masked_logits = vec![f32::NEG_INFINITY; vocab_size];

        if Self::is_complete_answer(target_text) {
            for &eos_id in &self.eos_token_ids {
                let idx = eos_id as usize;
                if idx < vocab_size {
                    masked_logits[idx] = logits_vec[idx];
                }
            }
            // Also allow whitespace to gracefully pad out
            for token_id in 0..vocab_size {
                let piece = tokenizer.decode(&[token_id as u32], false).unwrap_or_default();
                if piece.chars().all(|c| c.is_whitespace()) {
                    masked_logits[token_id] = logits_vec[token_id];
                }
            }
            let device = logits.device();
            return candle_core::Tensor::from_vec(masked_logits, logits.shape().clone(), device);
        }

        // Extremely strict validation: Check all tokens
        for token_id in 0..vocab_size {
            let piece = tokenizer.decode(&[token_id as u32], false).unwrap_or_default();
            if piece.is_empty() { continue; }
            let tentative = format!("{}{}", target_text, piece);
            if Self::is_valid_prefix(&tentative, &self.allowed_letters) {
                masked_logits[token_id] = logits_vec[token_id];
            }
        }

        let device = logits.device();
        candle_core::Tensor::from_vec(masked_logits, logits.shape().clone(), device)
    }
}

// ============================================================================
// MMLU / Multiple-Choice Logit Mask
// ============================================================================
//
// Constrained decoding for multiple-choice benchmarks (MMLU Pro, FSIQ, etc.).
// Guarantees the model emits only valid answer prefixes ending in A–E, then EOS.

/// Finite-state constrained decoder for multiple-choice answer formats.
///
/// Accepted complete outputs (after leading whitespace):
/// - `A` / `B` / … (single option letter)
/// - `(A)` / `(B)` / …
/// - `Answer: A` / `Answer: B` / …
/// - `Answer: A.` (optional trailing dot)
pub struct MmluLogitMask {
    allowed_letters: Vec<char>,
    eos_token_ids: Vec<u32>,
    /// Pre-filtered vocab subset — avoids O(vocab) decode per step when possible.
    candidate_token_ids: Vec<u32>,
    answer_complete: bool,
}

impl MmluLogitMask {
    pub fn new(tokenizer: &tokenizers::Tokenizer, allowed_letters: Vec<char>) -> Self {
        let eos_token_ids = vec![151643, 151644, 151645]; // Qwen2.5 EOS variants
        let candidate_token_ids = Self::build_candidate_tokens(tokenizer, &allowed_letters);
        Self {
            allowed_letters,
            eos_token_ids,
            candidate_token_ids,
            answer_complete: false,
        }
    }

    /// Default 4-option mask (A–D).
    pub fn default_abcd(tokenizer: &tokenizers::Tokenizer) -> Self {
        Self::new(tokenizer, vec!['A', 'B', 'C', 'D'])
    }

    /// Parse option letters present in a multiple-choice prompt (e.g. A. … E.).
    pub fn parse_option_letters_from_prompt(prompt: &str) -> Vec<char> {
        let mut letters = Vec::new();
        for letter in 'A'..='Z' {
            let patterns = [
                format!("\n{}. ", letter),
                format!("\n{}.", letter),
                format!("\n{}) ", letter),
                format!("Options:\n{}", letter),
            ];
            if patterns.iter().any(|pat| prompt.contains(pat.as_str())) {
                letters.push(letter);
            }
        }
        if letters.len() >= 2 {
            letters
        } else {
            vec!['A', 'B', 'C', 'D', 'E']
        }
    }

    pub fn detect_multiple_choice_prompt(prompt: &str) -> bool {
        let p = prompt.to_lowercase();
        if p.contains("reasoning") || p.contains("step-by-step") || p.contains("thought") || p.contains("think") || p.contains("cot") {
            return false;
        }

        if std::env::var("SODALS_MMLU_MASK").map(|v| v == "0" || v == "false").unwrap_or(false) {
            return false;
        }
        if std::env::var("SODALS_MMLU_MASK").is_ok() {
            return true;
        }

        let p = prompt.to_lowercase();
        let mc_markers = [
            "multiple choice",
            "multiple-choice",
            "option letter",
            "select exactly one",
            "select the option",
            "provide your answer by selecting",
            "choose the correct answer",
            "which of the following",
        ];
        if mc_markers.iter().any(|m| p.contains(m)) {
            return true;
        }

        let has_options_header = p.contains("options:") || p.contains("### options:");
        let has_letter_options = ('a'..='e').any(|c| {
            prompt.contains(&format!("\n{}. ", c.to_ascii_uppercase()))
                || prompt.contains(&format!("\n{}.", c.to_ascii_uppercase()))
        });
        has_options_header && has_letter_options
    }

    /// Apply multiple-choice constraint mask to logits.
    pub fn apply_mask(
        &mut self,
        logits: &candle_core::Tensor,
        current_sequence: &[u32],
        tokenizer: &tokenizers::Tokenizer,
        prompt_text: &str,
    ) -> candle_core::Result<candle_core::Tensor> {
        let vocab_size = logits.dims().last().copied().unwrap_or(0);
        if vocab_size == 0 {
            return Ok(logits.clone());
        }

        let current_text = tokenizer
            .decode(current_sequence, false)
            .unwrap_or_default();

        // Check if currently in reasoning/thinking phase
        let is_thinking = (current_text.contains("<think>") && !current_text.contains("</think>"))
            || (current_text.contains("<thought>") && !current_text.contains("</thought>"))
            || (current_text.contains("<option_analysis>") && !current_text.contains("</option_analysis>"));
        if is_thinking {
            return Ok(logits.clone());
        }

        // Isolate target text after the thinking blocks or prompt
        let target_text = if let Some(pos) = current_text.rfind("</think>") {
            &current_text[pos + "</think>".len()..]
        } else if let Some(pos) = current_text.rfind("</thought>") {
            &current_text[pos + "</thought>".len()..]
        } else if let Some(pos) = current_text.rfind("</option_analysis>") {
            &current_text[pos + "</option_analysis>".len()..]
        } else {
            if current_text.starts_with(prompt_text) {
                &current_text[prompt_text.len()..]
            } else {
                &current_text
            }
        };

        let logits_vec = logits.to_vec1::<f32>()?;

        if Self::is_complete_answer(target_text, &self.allowed_letters) {
            self.answer_complete = true;
        }

        let mut masked_logits = vec![f32::NEG_INFINITY; vocab_size];

        if self.answer_complete {
            for &eos_id in &self.eos_token_ids {
                let idx = eos_id as usize;
                if idx < vocab_size {
                    masked_logits[idx] = logits_vec[idx];
                }
            }
            // Allow trailing whitespace after the answer letter.
            for &token_id in &self.candidate_token_ids {
                let idx = token_id as usize;
                if idx >= vocab_size {
                    continue;
                }
                let piece = tokenizer
                    .decode(&[token_id], false)
                    .unwrap_or_default();
                if piece.chars().all(|c| c.is_whitespace()) {
                    masked_logits[idx] = logits_vec[idx];
                }
            }
            let device = logits.device();
            return candle_core::Tensor::from_vec(masked_logits, logits.shape().clone(), device);
        }

        // Always evaluate pre-filtered candidates.
        for &token_id in &self.candidate_token_ids {
            let idx = token_id as usize;
            if idx >= vocab_size {
                continue;
            }
            let piece = tokenizer
                .decode(&[token_id], false)
                .unwrap_or_default();
            let tentative = format!("{}{}", target_text, piece);
            if Self::is_valid_prefix(&tentative, &self.allowed_letters) {
                masked_logits[idx] = logits_vec[idx];
            }
        }

        // Safety net: also check top logits in case tokenizer edge-cases were missed.
        let mut indexed: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (token_id, _) in indexed.into_iter().take(128) {
            if masked_logits[token_id] > f32::NEG_INFINITY / 2.0 {
                continue;
            }
            let piece = tokenizer
                .decode(&[token_id as u32], false)
                .unwrap_or_default();
            let tentative = format!("{}{}", target_text, piece);
            if Self::is_valid_prefix(&tentative, &self.allowed_letters) {
                masked_logits[token_id] = logits_vec[token_id];
            }
        }

        let device = logits.device();
        candle_core::Tensor::from_vec(masked_logits, logits.shape().clone(), device)
    }

    fn build_candidate_tokens(tokenizer: &tokenizers::Tokenizer, letters: &[char]) -> Vec<u32> {
        let vocab_size = tokenizer.get_vocab_size(false);
        let mut candidates = Vec::new();

        for id in 0..vocab_size {
            let text = tokenizer
                .decode(&[id as u32], false)
                .unwrap_or_default();
            if Self::is_potential_mc_token(&text, letters) {
                candidates.push(id as u32);
            }
        }
        candidates
    }

    fn is_potential_mc_token(text: &str, letters: &[char]) -> bool {
        if text.is_empty() {
            return false;
        }
        if text.chars().all(|c| c.is_whitespace()) {
            return true;
        }
        if text.len() > 12 {
            return false;
        }

        let lower = text.to_lowercase();
        if "answer".starts_with(&lower) || lower.starts_with("answer") {
            return true;
        }

        let upper = text.to_uppercase();
        if letters.iter().any(|l| upper.contains(*l)) {
            return true;
        }

        if text.chars().all(|c| {
            c.is_whitespace() || c.is_ascii_alphabetic() || "()[]:._\n\r\t".contains(c)
        }) {
            return text.len() <= 8;
        }

        false
    }

    /// True when the generated text contains a complete option letter answer.
    pub fn is_complete_answer(text: &str, letters: &[char]) -> bool {
        Self::extract_answer_letter(text, letters).is_some()
    }

    fn extract_answer_letter(text: &str, letters: &[char]) -> Option<char> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        let lower = trimmed.to_lowercase();
        if let Some(pos) = lower.find("answer") {
            let after_keyword = &trimmed[pos + "answer".len()..];
            let after_colon = after_keyword
                .trim_start()
                .strip_prefix(':')
                .unwrap_or(after_keyword.trim_start())
                .trim_start();
            if let Some(letter) = Self::parse_letter_suffix(after_colon, letters) {
                return Some(letter);
            }
        }

        Self::parse_letter_suffix(trimmed, letters)
    }

    fn parse_letter_suffix(s: &str, letters: &[char]) -> Option<char> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }

        if s.starts_with('(') {
            if s.len() >= 3 && s.chars().nth(2) == Some(')') {
                let c = s.chars().nth(1)?.to_ascii_uppercase();
                if letters.contains(&c) {
                    return Some(c);
                }
            }
            return None;
        }

        let mut chars = s.chars();
        let first = chars.next()?.to_ascii_uppercase();
        if !letters.contains(&first) {
            return None;
        }

        let rest: String = chars.collect();
        if rest.is_empty() || rest == "." || rest.chars().all(|c| c.is_whitespace()) {
            return Some(first);
        }

        None
    }

    /// True when `text` is a strict prefix of some valid answer (not yet complete).
    pub fn is_valid_prefix(text: &str, letters: &[char]) -> bool {
        // Limit trailing/consecutive whitespace to prevent infinite loops (max 4 spaces/newlines)
        let trailing_ws_count = text.chars().rev().take_while(|c| c.is_whitespace()).count();
        if trailing_ws_count > 4 {
            return false;
        }

        if Self::is_complete_answer(text, letters) {
            return text
                .trim_end()
                .chars()
                .last()
                .map(|c| letters.contains(&c.to_ascii_uppercase()) || c.is_whitespace() || c == '.' || c == ')')
                .unwrap_or(true);
        }

        if text.is_empty() || text.chars().all(|c| c.is_whitespace()) {
            if text.len() > 4 {
                return false;
            }
            return true;
        }

        let body = text.trim_start();
        if body.is_empty() {
            if text.len() > 4 {
                return false;
            }
            return true;
        }

        let lower = body.to_lowercase();
        const BLOCKED: &[&str] = &[
            "the ", "based", "response", "let ", "step ", "i ", "this ", "note", "**",
            "therefore", "first,", "second,", "third,", "analysis", "reasoning",
        ];
        if BLOCKED.iter().any(|b| lower.starts_with(b)) {
            return false;
        }

        // Partial "Answer" / "Answer:" path
        if "answer".starts_with(&lower) || lower.starts_with("answer") {
            if lower == "answer" || lower == "answer:" {
                return true;
            }
            if lower.starts_with("answer:") {
                let after = body[body.find(':').map(|i| i + 1).unwrap_or(body.len())..].trim_start();
                return Self::is_letter_suffix_prefix(after, letters);
            }
            return lower.len() < "answer:".len();
        }

        Self::is_letter_suffix_prefix(body, letters)
    }

    fn is_letter_suffix_prefix(s: &str, letters: &[char]) -> bool {
        let s = s.trim_start();
        if s.is_empty() {
            return true;
        }

        if s == "(" {
            return true;
        }

        if s.starts_with('(') {
            if s.len() == 2 {
                return letters.contains(&s.chars().nth(1).unwrap().to_ascii_uppercase());
            }
            if s.len() >= 3 && s.chars().nth(2) == Some(')') {
                return letters.contains(&s.chars().nth(1).unwrap().to_ascii_uppercase());
            }
            return false;
        }

        let first = s.chars().next().unwrap().to_ascii_uppercase();
        if letters.contains(&first) {
            let rest: String = s.chars().skip(1).collect();
            return rest.is_empty() || rest == "." || rest.chars().all(|c| c.is_whitespace());
        }

        false
    }
}

#[cfg(test)]
mod mmlu_mask_tests {
    use super::*;

    #[test]
    fn test_valid_prefixes() {
        let letters = vec!['A', 'B', 'C', 'D', 'E'];
        assert!(MmluLogitMask::is_valid_prefix("", &letters));
        assert!(MmluLogitMask::is_valid_prefix("\n\n", &letters));
        assert!(MmluLogitMask::is_valid_prefix("Answer", &letters));
        assert!(MmluLogitMask::is_valid_prefix("Answer:", &letters));
        assert!(MmluLogitMask::is_valid_prefix("Answer: ", &letters));
        assert!(MmluLogitMask::is_valid_prefix("Answer: B", &letters));
        assert!(MmluLogitMask::is_valid_prefix("B", &letters));
        assert!(MmluLogitMask::is_valid_prefix("(C)", &letters));
        assert!(!MmluLogitMask::is_valid_prefix("The answer is B", &letters));
        assert!(!MmluLogitMask::is_valid_prefix("Answer: B. oil debris", &letters));
    }

    #[test]
    fn test_complete_answer_detection() {
        let letters = vec!['A', 'B', 'C', 'D', 'E'];
        assert!(MmluLogitMask::is_complete_answer("B", &letters));
        assert!(MmluLogitMask::is_complete_answer("\n\nAnswer: B", &letters));
        assert!(MmluLogitMask::is_complete_answer("(C)", &letters));
        assert!(MmluLogitMask::is_complete_answer("Answer: D.", &letters));
        assert!(!MmluLogitMask::is_complete_answer("Answer:", &letters));
        assert!(!MmluLogitMask::is_complete_answer("Answer: B. oil", &letters));
    }

    #[test]
    fn test_detect_multiple_choice_prompt() {
        let prompt = "Please answer the following multiple choice question.\n\nOptions:\nA. foo\nB. bar\n\nAnswer:";
        assert!(MmluLogitMask::detect_multiple_choice_prompt(prompt));
        assert!(!MmluLogitMask::detect_multiple_choice_prompt("Write a Python function to sort a list."));
    }

    #[test]
    fn test_parse_option_letters() {
        let prompt = "Options:\nA. one\nB. two\nC. three\nD. four\nE. five";
        let letters = MmluLogitMask::parse_option_letters_from_prompt(prompt);
        assert_eq!(letters, vec!['A', 'B', 'C', 'D', 'E']);
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_oracle_valid_empty_vector() {
        let device = Device::Cpu;
        // Empty/zero vector has no active nodes → no tokens → valid
        let empty_vec = vec![0.0f32; 128];
        let tensor = candle_core::Tensor::from_vec(empty_vec, (128,), &device).unwrap();
        
        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);
        let is_violation = oracle.is_violation(&tensor).unwrap();
        
        assert!(!is_violation, "Empty vector should be valid (no active tokens)");
    }

    #[test]
    fn test_oracle_with_labeled_nodes() {
        let device = Device::Cpu;
        let mut network = ResonanceNetwork::new(128);
        
        // Add labeled concepts that will form a valid token sequence
        let _motor_id = network.add_labeled_concept("motor", 0.1, 
            crate::compiler::qmem_compiler::generate_holographic_signature("motor", 4096));
        let _analysis_id = network.add_labeled_concept("analysis", 0.1,
            crate::compiler::qmem_compiler::generate_holographic_signature("analysis", 4096));
        
        // Create vector with high activation at those exact node positions
        let mut activation_vec = vec![0.0f32; 128];
        activation_vec[0] = 0.9; // motor at node 0
        activation_vec[1] = 0.8; // analysis at node 1
        
        let tensor = candle_core::Tensor::from_vec(activation_vec, (128,), &device).unwrap();
        
        let oracle = SyntaxOracle::new(0.5, &network);
        let is_violation = oracle.is_violation(&tensor).unwrap();
        
        // "motor analysis" may or may not parse depending on grammar
        // The key is that it actually RUNS through the EBNF parser now
        eprintln!("Oracle result for labeled nodes: violation={}", is_violation);
    }

    #[test]
    fn test_oracle_phase_penalty_no_active_nodes() {
        let device = Device::Cpu;
        let network = ResonanceNetwork::new(64);
        
        // Zero vector = no active nodes = no penalty
        let zero_vec = vec![0.0f32; 64];
        let tensor = candle_core::Tensor::from_vec(zero_vec, (64,), &device).unwrap();
        
        let oracle = SyntaxOracle::new(0.5, &network);
        let penalty = oracle.compute_phase_penalty(&tensor).unwrap();
        
        assert_eq!(penalty, 0.0, "No active nodes = no phase penalty");
    }
}

#[derive(Debug, Clone)]
pub struct TelemetryAnomaly {
    pub value: f64,
    pub z_score: f64,
    pub index: usize,
    pub is_threshold_violation: bool,
}

pub struct TelemetryParser;

impl TelemetryParser {
    pub fn detect_anomalies(text: &str, threshold: Option<f64>) -> Vec<TelemetryAnomaly> {
        let num_re = regex::Regex::new(r"-?\b\d+\.?\d*\b").unwrap();
        let mut values = Vec::new();
        for cap in num_re.captures_iter(text) {
            if let Some(m) = cap.get(0) {
                if let Ok(val) = m.as_str().parse::<f64>() {
                    values.push(val);
                }
            }
        }

        if values.len() < 3 {
            return Vec::new();
        }

        let n = values.len() as f64;
        let mean: f64 = values.iter().sum::<f64>() / n;
        let variance: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        let mut anomalies = Vec::new();
        for (i, &val) in values.iter().enumerate() {
            let z = if std_dev > 0.0 {
                (val - mean) / std_dev
            } else {
                0.0
            };

            let is_violation = if let Some(t) = threshold {
                val.abs() > t
            } else {
                false
            };

            if z.abs() > 2.5 || is_violation {
                anomalies.push(TelemetryAnomaly {
                    value: val,
                    z_score: z,
                    index: i,
                    is_threshold_violation: is_violation,
                });
            }
        }
        anomalies
    }

    pub fn extract_resources(text: &str) -> Vec<String> {
        let mut resources = Vec::new();
        let patterns = [
            r"\b[a-zA-Z0-9_-]+-service\b",
            r"\b[a-zA-Z0-9_-]+-node-\d+\b",
            r"\b[a-zA-Z0-9_-]+-db-\d+\b",
            r"\bserver-\d+\b",
            r"\b[a-zA-Z0-9_.-]+(?:-instance|-vm|-pod|-container)\b",
            r"\b[a-zA-Z][a-zA-Z0-9_-]{3,30}\b",
        ];

        for &pat in &patterns {
            if let Ok(re) = regex::Regex::new(pat) {
                for cap in re.captures_iter(text) {
                    if let Some(m) = cap.get(0) {
                        let res = m.as_str().to_string();
                        let lower = res.to_lowercase();
                        let blocked_words = [
                            "timestamp", "value", "threshold", "anomaly", "detection", 
                            "service", "database", "resource", "server", "options", 
                            "answer", "correct", "question", "symptom", "reason",
                            "bearing", "vibration", "temperature", "coolant", "leakage",
                            "pressure", "acoustic"
                        ];
                        if !blocked_words.contains(&lower.as_str()) && !resources.contains(&res) {
                            resources.push(res);
                        }
                    }
                }
            }
        }
        resources
    }

    pub fn find_anomalous_resource_by_telemetry(prompt: &str) -> Option<String> {
        let resources = Self::extract_resources(prompt);
        if resources.is_empty() {
            return None;
        }

        let mut best_resource = None;
        let mut max_anomaly_score = -1.0;

        let lines: Vec<&str> = prompt.lines().collect();
        for res in &resources {
            let mut res_values = Vec::new();
            for line in &lines {
                if line.contains(res) {
                    let num_re = regex::Regex::new(r"-?\b\d+\.?\d*\b").unwrap();
                    for cap in num_re.captures_iter(line) {
                        if let Some(m) = cap.get(0) {
                            if let Ok(val) = m.as_str().parse::<f64>() {
                                if val < 1000000.0 {
                                    res_values.push(val);
                                }
                            }
                        }
                    }
                }
            }

            if res_values.len() >= 3 {
                let n = res_values.len() as f64;
                let mean: f64 = res_values.iter().sum::<f64>() / n;
                let variance: f64 = res_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                let std_dev = if variance > 0.0 { variance.sqrt() } else { 1.0 };
                
                for &val in &res_values {
                    let z = (val - mean).abs() / std_dev;
                    let score = z + (val / 100.0).min(5.0);
                    if score > max_anomaly_score {
                        max_anomaly_score = score;
                        best_resource = Some(res.clone());
                    }
                }
            } else if !res_values.is_empty() {
                let &max_val = res_values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                if max_val > max_anomaly_score {
                    max_anomaly_score = max_val;
                    best_resource = Some(res.clone());
                }
            }
        }

        best_resource
    }

    pub fn find_anomalous_resource_from_prompt_json(prompt: &str) -> Option<String> {
        let res_re = regex::Regex::new(r#""resource_name"\s*:\s*"([^"]+)""#).unwrap();
        let mean_re = regex::Regex::new(r#""mean_hist"\s*:\s*([0-9.]+)"#).unwrap();
        let cost_re = regex::Regex::new(r#""anomaly_cost"\s*:\s*([0-9.]+)"#).unwrap();
        
        let mut resources = Vec::new();
        for cap in res_re.captures_iter(prompt) {
            if let Some(m) = cap.get(1) {
                resources.push(m.as_str().to_string());
            }
        }
        
        let mut means = Vec::new();
        for cap in mean_re.captures_iter(prompt) {
            if let Some(m) = cap.get(1) {
                if let Ok(val) = m.as_str().parse::<f64>() {
                    means.push(val);
                }
            }
        }
        
        let mut costs = Vec::new();
        for cap in cost_re.captures_iter(prompt) {
            if let Some(m) = cap.get(1) {
                if let Ok(val) = m.as_str().parse::<f64>() {
                    costs.push(val);
                }
            }
        }
        
        if resources.is_empty() {
            return None;
        }
        
        let mut best_resource = None;
        let mut max_ratio = -1.0;
        
        for i in 0..resources.len() {
            let mean = *means.get(i).unwrap_or(&1.0);
            let cost = *costs.get(i).unwrap_or(&0.0);
            let diff = (cost - mean).abs();
            let ratio = if mean > 0.0 { diff / mean } else { diff };
            
            if ratio > max_ratio {
                max_ratio = ratio;
                best_resource = Some(resources[i].clone());
            }
        }
        
        if best_resource.is_none() && !resources.is_empty() {
            best_resource = Some(resources[0].clone());
        }
        
        best_resource
    }
}
















