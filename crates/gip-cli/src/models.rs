//! Models in the Hugging Face cache: names, downloads, listing, and removal.
//!
//! A model name is a Hugging Face repository with an optional quantization tag, as in
//! `LiquidAI/LFM2.5-2.6B-GGUF:Q8_0`. The tag picks the GGUF file whose name ends in `-TAG.gguf`.
//! Files live in the shared cache at `~/.cache/huggingface/hub`, which other Hugging Face tools
//! read too, so no model downloads twice.

use std::fmt;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use hf_hub::HFClientSync;
use hf_hub::cache::CachedFileInfo;
use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};
use hf_hub::repository::RepoTreeEntry;

use crate::Error;

/// The tag a name without one picks: the only quantization gip's Metal path runs today.
const DEFAULT_TAG: &str = "Q8_0";

/// A model name: a Hugging Face repository and an optional quantization tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Name {
    owner: String,
    repo: String,
    tag: Option<String>,
}

impl Name {
    /// Parse `text` as `owner/repo` or `owner/repo:tag`.
    pub(crate) fn parse(text: &str) -> Result<Self, Error> {
        let (id, tag) = match text.split_once(':') {
            Some((id, tag)) => (id, Some(tag.to_owned())),
            None => (text, None),
        };
        let invalid = || format!("{text} is no model name like LiquidAI/LFM2.5-2.6B-GGUF:Q8_0");
        let (owner, repo) = id.split_once('/').ok_or_else(invalid)?;
        if owner.is_empty() || repo.is_empty() || repo.contains('/') || tag.as_deref() == Some("") {
            return Err(invalid().into());
        }
        Ok(Self {
            owner: owner.to_owned(),
            repo: repo.to_owned(),
            tag,
        })
    }

    fn repo_id(&self) -> String {
        format!("{}/{}", self.owner, self.repo)
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.owner, self.repo)?;
        if let Some(tag) = &self.tag {
            write!(f, ":{tag}")?;
        }
        Ok(())
    }
}

/// Return the GGUF file among `files` that `tag` names.
///
/// With no tag, a repository holding one GGUF file yields that file, and any other yields its Q8_0
/// file. Several files can end in the same tag, such as `M-Q4_0.gguf` and `M-QAD-Q4_0.gguf`, and
/// the shortest name wins, so `:QAD-Q4_0` still reaches the longer one.
fn select<'f>(files: &[&'f str], tag: Option<&str>) -> Option<&'f str> {
    let ggufs: Vec<&str> = files
        .iter()
        .copied()
        .filter(|file| !file.contains('/') && file.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    let tag = match (tag, ggufs.as_slice()) {
        (None, [only]) => return Some(only),
        (None, _) => DEFAULT_TAG,
        (Some(tag), _) => tag,
    };
    let suffix = format!("-{tag}.gguf").to_ascii_lowercase();
    ggufs
        .into_iter()
        .filter(|file| file.to_ascii_lowercase().ends_with(&suffix))
        .min_by_key(|file| file.len())
}

/// Return the shortest tag that selects `file` among `files`.
fn tag_of(file: &str, files: &[&str]) -> String {
    let stem = file.strip_suffix(".gguf").unwrap_or(file);
    let segments: Vec<&str> = stem.split('-').collect();
    (1..=segments.len())
        .map(|count| segments[segments.len() - count..].join("-"))
        .find(|tag| select(files, Some(tag)) == Some(file))
        .unwrap_or_else(|| stem.to_owned())
}

/// Return the GGUF file for `model`, which is a path to a file or a model name. A name not yet in
/// the cache downloads first.
pub(crate) fn resolve(model: &str) -> Result<PathBuf, Error> {
    let path = Path::new(model);
    if path.exists() {
        return Ok(path.to_owned());
    }
    let name = Name::parse(model)
        .map_err(|_| format!("{model} is neither a file nor a model name like owner/repo:Q8_0"))?;
    match cached(&name)? {
        Some(file) => Ok(file.file_path),
        None => pull(&name),
    }
}

/// Download the GGUF file `name` picks into the cache and return its path.
pub(crate) fn pull(name: &Name) -> Result<PathBuf, Error> {
    let client = HFClientSync::new()?;
    let repo = client.model(&name.owner, &name.repo);
    let entries = repo.list_tree().send()?;
    let files: Vec<&str> = entries
        .iter()
        .filter_map(|entry| match entry {
            RepoTreeEntry::File { path, .. } => Some(path.as_str()),
            RepoTreeEntry::Directory { .. } => None,
        })
        .collect();
    let file = select(&files, name.tag.as_deref()).ok_or_else(|| missing(name, &files))?;
    let path = repo
        .download_file()
        .filename(file)
        .progress(Progress::new(ProgressLine::new(format!(
            "{}:{}",
            name.repo_id(),
            tag_of(file, &files)
        ))))
        .send()?;
    Ok(path)
}

/// Return the error for a repository that holds no file `name` picks, listing the tags it holds.
fn missing(name: &Name, files: &[&str]) -> Error {
    let tags: Vec<String> = files
        .iter()
        .filter(|file| select(files, Some(&tag_of(file, files))) == Some(file))
        .map(|file| tag_of(file, files))
        .collect();
    let tag = name.tag.as_deref().unwrap_or(DEFAULT_TAG);
    if tags.is_empty() {
        format!("{} holds no GGUF files", name.repo_id()).into()
    } else {
        format!(
            "{} has no {tag} file, only {}",
            name.repo_id(),
            tags.join(", ")
        )
        .into()
    }
}

/// Return every cached GGUF file of the repository `repo_id`, from the revision `main` names when
/// the cache holds it.
fn cached_files(repo_id: &str) -> Result<Vec<CachedFileInfo>, Error> {
    let cache = HFClientSync::new()?.scan_cache().send()?;
    let Some(repo) = cache
        .repos
        .into_iter()
        .find(|repo| repo.repo_type == "model" && repo.repo_id == repo_id)
    else {
        return Ok(Vec::new());
    };
    let mut revisions = repo.revisions;
    revisions.sort_by_key(|revision| !revision.refs.iter().any(|name| name == "main"));
    let mut files: Vec<CachedFileInfo> = Vec::new();
    for revision in revisions {
        for file in revision.files {
            if !files.iter().any(|seen| seen.file_name == file.file_name) {
                files.push(file);
            }
        }
    }
    Ok(files)
}

/// Return the cached file `name` picks, if the cache holds it.
fn cached(name: &Name) -> Result<Option<CachedFileInfo>, Error> {
    let files = cached_files(&name.repo_id())?;
    let names: Vec<&str> = files.iter().map(|file| file.file_name.as_str()).collect();
    let Some(chosen) = select(&names, name.tag.as_deref()) else {
        return Ok(None);
    };
    Ok(files.iter().find(|file| file.file_name == chosen).cloned())
}

/// Return every cached GGUF model as its name and size in bytes.
pub(crate) fn list() -> Result<Vec<(String, u64)>, Error> {
    let cache = HFClientSync::new()?.scan_cache().send()?;
    let mut models = Vec::new();
    for repo in cache.repos.iter().filter(|repo| repo.repo_type == "model") {
        let files = cached_files(&repo.repo_id)?;
        let names: Vec<&str> = files.iter().map(|file| file.file_name.as_str()).collect();
        for file in &files {
            if select(&names, Some(&tag_of(&file.file_name, &names)))
                == Some(file.file_name.as_str())
            {
                let tag = tag_of(&file.file_name, &names);
                models.push((format!("{}:{tag}", repo.repo_id), file.size_on_disk));
            }
        }
    }
    models.sort();
    Ok(models)
}

/// Remove the cached file `name` picks. The file's data goes too, unless another cached file
/// shares it.
pub(crate) fn remove(name: &Name) -> Result<(), Error> {
    let file = cached(name)?.ok_or_else(|| format!("the cache holds no {name}"))?;
    fs::remove_file(&file.file_path)?;
    let cache = HFClientSync::new()?.scan_cache().send()?;
    let shared = cache.repos.iter().any(|repo| {
        repo.revisions
            .iter()
            .flat_map(|revision| &revision.files)
            .any(|other| other.blob_path == file.blob_path)
    });
    if !shared {
        fs::remove_file(&file.blob_path)?;
    }
    Ok(())
}

/// Return `bytes` in the largest decimal unit that keeps the number at 1 or more.
pub(crate) fn human_size(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1000.0 && unit + 1 < units.len() {
        size /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {}", units[unit])
    }
}

/// A download progress line on standard error, shown only on a terminal.
struct ProgressLine {
    label: String,
    completed: AtomicU64,
    total: AtomicU64,
}

impl ProgressLine {
    fn new(label: String) -> Self {
        Self {
            label,
            completed: AtomicU64::new(0),
            total: AtomicU64::new(0),
        }
    }
}

impl ProgressHandler for ProgressLine {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                self.total.store(*total_bytes, Ordering::Relaxed);
            }
            DownloadEvent::Progress { files } => {
                let completed: u64 = files.iter().map(|file| file.bytes_completed).sum();
                self.completed.fetch_max(completed, Ordering::Relaxed);
            }
            DownloadEvent::AggregateProgress {
                bytes_completed, ..
            } => {
                self.completed
                    .fetch_max(*bytes_completed, Ordering::Relaxed);
            }
            DownloadEvent::Complete => {
                let total = self.total.load(Ordering::Relaxed);
                self.completed.store(total, Ordering::Relaxed);
            }
        }
        let mut stderr = io::stderr().lock();
        if !stderr.is_terminal() {
            return;
        }
        let total = self.total.load(Ordering::Relaxed).max(1);
        let completed = self.completed.load(Ordering::Relaxed).min(total);
        let percent = completed * 100 / total;
        // A failed progress write leaves the download itself unaffected.
        let _ = write!(
            stderr,
            "\rpulling {} {percent:3}% of {}",
            self.label,
            human_size(total)
        );
        if matches!(event, DownloadEvent::Complete) {
            let _ = writeln!(stderr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILES: &[&str] = &[
        "README.md",
        "LFM2.5-2.6B-Q4_0.gguf",
        "LFM2.5-2.6B-QAD-Q4_0.gguf",
        "LFM2.5-2.6B-Q8_0.gguf",
        "qad/model.gguf",
    ];

    #[test]
    fn tags_pick_files() {
        assert_eq!(select(FILES, None), Some("LFM2.5-2.6B-Q8_0.gguf"));
        assert_eq!(select(FILES, Some("q4_0")), Some("LFM2.5-2.6B-Q4_0.gguf"));
        assert_eq!(
            select(FILES, Some("QAD-Q4_0")),
            Some("LFM2.5-2.6B-QAD-Q4_0.gguf")
        );
        assert_eq!(select(FILES, Some("Q6_K")), None);
        assert_eq!(tag_of("LFM2.5-2.6B-QAD-Q4_0.gguf", FILES), "QAD-Q4_0");
        assert_eq!(select(&["only.gguf"], None), Some("only.gguf"));
    }

    #[test]
    fn names_parse() {
        let name = Name::parse("LiquidAI/LFM2.5-2.6B-GGUF:Q8_0").unwrap();
        assert_eq!(name.to_string(), "LiquidAI/LFM2.5-2.6B-GGUF:Q8_0");
        assert!(Name::parse("models/x.gguf/extra").is_err());
        assert!(Name::parse("LiquidAI/LFM2.5-2.6B-GGUF:").is_err());
    }
}
