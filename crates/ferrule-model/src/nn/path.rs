use std::fmt;
use std::str::FromStr;

use thiserror::Error;

/// Canonical dot-separated path of a module or parameter.
///
/// Components are limited to ASCII letters, digits, and `_`. The empty path is
/// reserved for the root module and can only be constructed with [`Self::root`].
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModulePath(String);

impl ModulePath {
    /// Returns the root module path.
    pub const fn root() -> Self {
        Self(String::new())
    }

    /// Parses a non-root canonical path.
    pub fn new(path: impl Into<String>) -> Result<Self, ModulePathError> {
        let path = path.into();
        validate_path(&path)?;
        Ok(Self(path))
    }

    /// Returns the canonical string. The root path is represented by `""`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    /// Appends one validated path component.
    pub fn child(&self, component: &str) -> Result<Self, ModulePathError> {
        validate_component(component, component)?;
        if self.is_root() {
            Self::new(component)
        } else {
            Self::new(format!("{}.{}", self.0, component))
        }
    }

    /// Returns the direct parent, including the root parent of a top-level path.
    pub fn parent(&self) -> Option<Self> {
        if self.is_root() {
            return None;
        }
        match self.0.rsplit_once('.') {
            Some((parent, _)) => Some(Self(parent.to_string())),
            None => Some(Self::root()),
        }
    }

    pub fn name(&self) -> Option<&str> {
        if self.is_root() {
            None
        } else {
            Some(self.0.rsplit('.').next().expect("non-root path has a name"))
        }
    }

    pub fn segments(&self) -> impl DoubleEndedIterator<Item = &str> {
        self.0.split('.').filter(|segment| !segment.is_empty())
    }

    pub fn is_direct_child_of(&self, parent: &Self) -> bool {
        self.parent().as_ref() == Some(parent)
    }
}

impl fmt::Display for ModulePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ModulePath {
    type Err = ModulePathError;

    fn from_str(path: &str) -> Result<Self, Self::Err> {
        Self::new(path)
    }
}

impl TryFrom<String> for ModulePath {
    type Error = ModulePathError;

    fn try_from(path: String) -> Result<Self, Self::Error> {
        Self::new(path)
    }
}

impl TryFrom<&str> for ModulePath {
    type Error = ModulePathError;

    fn try_from(path: &str) -> Result<Self, Self::Error> {
        Self::new(path)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ModulePathError {
    #[error("the empty canonical path is reserved for the root module")]
    Empty,
    #[error("canonical path '{path}' contains an empty component")]
    EmptyComponent { path: String },
    #[error("canonical path '{path}' has invalid component '{component}'")]
    InvalidComponent { path: String, component: String },
}

fn validate_path(path: &str) -> Result<(), ModulePathError> {
    if path.is_empty() {
        return Err(ModulePathError::Empty);
    }
    for component in path.split('.') {
        if component.is_empty() {
            return Err(ModulePathError::EmptyComponent {
                path: path.to_string(),
            });
        }
        validate_component(component, path)?;
    }
    Ok(())
}

fn validate_component(component: &str, path: &str) -> Result<(), ModulePathError> {
    if component.is_empty() {
        return Err(ModulePathError::EmptyComponent {
            path: path.to_string(),
        });
    }
    if component
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        Ok(())
    } else {
        Err(ModulePathError::InvalidComponent {
            path: path.to_string(),
            component: component.to_string(),
        })
    }
}
