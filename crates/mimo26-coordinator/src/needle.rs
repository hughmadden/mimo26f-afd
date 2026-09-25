//! X1a needle prompt (ADVISOR-I4 §3.0) — the deterministic prompt whose
//! rendered form must match `rendered_sha256=95e240d3…` (the spike receipt
//! `runs/20260923-i4/X1a-receipt.md`). Twin of `spike/needle_prompt.py`.
//!
//! One planted fact near token ~100, the verbatim-recall question near token
//! ~4,000, separated by deterministic dull filler. Expected clean output:
//! `KESTREL-41` then EOS (greedy `[3390, 784, 49051, 12, 19, 16, 151645]`).

pub const FACT_CODE: &str = "KESTREL-41";
pub const FACT: &str = "The access code for the north gate is KESTREL-41.";
pub const QUESTION: &str = "What is the access code for the north gate? Answer with only the code.";

const FILLER: [&str; 10] = [
    "The committee reviewed the quarterly paperwork and filed it in the usual cabinet.",
    "A meeting on Tuesday covered the budget for office supplies and nothing else.",
    "The minutes from the previous session were approved without amendments.",
    "Staff rotated the storage room labels and recorded the change in the ledger.",
    "The maintenance log lists three filter replacements and one door adjustment.",
    "Everyone agreed the hallway painting can wait until the next fiscal cycle.",
    "The supply order included paper clips, folders, and two boxes of staples.",
    "Attendance was noted for the record and the session was adjourned on time.",
    "The archive team indexed the old folders by date and by department code.",
    "A reminder was posted about the annual equipment inventory next month.",
];

/// Deterministic dull document `i` (1-based); the pool cycles, the index persists.
pub fn filler_doc(i: usize) -> String {
    format!("Document {i}: {}", FILLER[(i - 1) % FILLER.len()])
}

/// Pure-text build: head filler, FACT, body filler, QUESTION (space-joined).
pub fn build_text(head: usize, body: usize) -> String {
    let mut parts: Vec<String> = (0..head).map(|i| filler_doc(1 + i)).collect();
    parts.push(FACT.to_string());
    parts.extend((0..body).map(|j| filler_doc(1 + head + j)));
    parts.push(QUESTION.to_string());
    parts.join(" ")
}

/// Token-accurate placement: grow the filler until the FACT starts at/after
/// `fact_at` tokens and the QUESTION at/after `question_at` (acceptance `±tol`).
/// `encode` is `str -> token count`. Returns the raw text + placement meta.
pub fn place<F: Fn(&str) -> usize>(
    encode: F,
    fact_at: usize,
    question_at: usize,
    tol: usize,
) -> Result<(String, NeedleMeta), String> {
    let mut head = 0usize;
    let mut fact_start = 0usize;
    while fact_start < fact_at {
        head += 1;
        let prefix: Vec<String> = (0..head).map(|i| filler_doc(i + 1)).collect();
        fact_start = encode(&prefix.join(" "));
    }
    let mut prefix: Vec<String> = (0..head).map(|i| filler_doc(i + 1)).collect();
    prefix.push(FACT.to_string());
    let prefix = prefix.join(" ");

    let mut body = 0usize;
    let mut question_start = 0usize;
    while question_start < question_at {
        body += 1;
        let mut full = prefix.clone();
        full.push(' ');
        full.push_str(&(0..body).map(|j| filler_doc(head + 1 + j)).collect::<Vec<_>>().join(" "));
        question_start = encode(&full);
    }
    let text = build_text(head, body);
    if fact_start < fact_at.saturating_sub(tol) || fact_start > fact_at + tol {
        return Err(format!("fact placement drift: {fact_start} vs {fact_at}±{tol}"));
    }
    if question_start < question_at.saturating_sub(tol) || question_start > question_at + tol {
        return Err(format!("question placement drift: {question_start} vs {question_at}±{tol}"));
    }
    Ok((text, NeedleMeta { head_docs: head, body_docs: body, fact_start, question_start }))
}

#[derive(Debug, Clone, Copy)]
pub struct NeedleMeta {
    pub head_docs: usize,
    pub body_docs: usize,
    pub fact_start: usize,
    pub question_start: usize,
}
