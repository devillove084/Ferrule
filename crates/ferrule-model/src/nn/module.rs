use thiserror::Error;

use super::{ModulePath, ParameterId, ParameterSpec};

/// Fallible callbacks for deterministic pre-order module traversal.
///
/// A module is entered, its parameters are visited in registration order, its
/// child modules are recursively visited in registration order, and it is then
/// exited.
pub trait ModuleVisitor {
    fn enter_module(&mut self, _path: &ModulePath, _kind: &str) -> Result<(), ModuleVisitError> {
        Ok(())
    }

    fn visit_parameter(&mut self, _parameter: &ParameterSpec) -> Result<(), ModuleVisitError> {
        Ok(())
    }

    fn exit_module(&mut self, _path: &ModulePath, _kind: &str) -> Result<(), ModuleVisitError> {
        Ok(())
    }
}

/// Object-safe inference module interface.
pub trait Module: Send + Sync {
    fn path(&self) -> &ModulePath;
    fn kind(&self) -> &str;
    fn visit(&self, visitor: &mut dyn ModuleVisitor) -> Result<(), ModuleVisitError>;
}

/// Concrete ordered module tree useful to recipes and schema tooling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleNode {
    path: ModulePath,
    kind: String,
    parameters: Vec<ParameterSpec>,
    children: Vec<ModuleNode>,
}

impl ModuleNode {
    pub fn new(path: ModulePath, kind: impl Into<String>) -> Result<Self, ModuleTreeError> {
        let kind = kind.into();
        if kind.trim().is_empty() {
            return Err(ModuleTreeError::EmptyKind { path });
        }
        Ok(Self {
            path,
            kind,
            parameters: Vec::new(),
            children: Vec::new(),
        })
    }

    pub fn parameters(&self) -> &[ParameterSpec] {
        &self.parameters
    }

    pub fn children(&self) -> &[ModuleNode] {
        &self.children
    }

    pub fn add_parameter(
        &mut self,
        parameter: ParameterSpec,
    ) -> Result<&mut Self, ModuleTreeError> {
        if !parameter.path().is_direct_child_of(&self.path) {
            return Err(ModuleTreeError::ParameterParentMismatch {
                module: self.path.clone(),
                parameter: parameter.path().clone(),
            });
        }
        if self
            .parameters
            .iter()
            .any(|current| current.path() == parameter.path())
        {
            return Err(ModuleTreeError::DuplicateParameterPath {
                path: parameter.path().clone(),
            });
        }
        if self
            .parameters
            .iter()
            .any(|current| current.id() == parameter.id())
        {
            return Err(ModuleTreeError::DuplicateParameterId { id: parameter.id() });
        }
        self.parameters.push(parameter);
        Ok(self)
    }

    pub fn add_child(&mut self, child: ModuleNode) -> Result<&mut Self, ModuleTreeError> {
        if !child.path.is_direct_child_of(&self.path) {
            return Err(ModuleTreeError::ModuleParentMismatch {
                parent: self.path.clone(),
                child: child.path,
            });
        }
        if self
            .children
            .iter()
            .any(|current| current.path == child.path)
        {
            return Err(ModuleTreeError::DuplicateModulePath { path: child.path });
        }
        self.children.push(child);
        Ok(self)
    }
}

impl Module for ModuleNode {
    fn path(&self) -> &ModulePath {
        &self.path
    }

    fn kind(&self) -> &str {
        &self.kind
    }

    fn visit(&self, visitor: &mut dyn ModuleVisitor) -> Result<(), ModuleVisitError> {
        visitor.enter_module(&self.path, &self.kind)?;
        for parameter in &self.parameters {
            visitor.visit_parameter(parameter)?;
        }
        for child in &self.children {
            child.visit(visitor)?;
        }
        visitor.exit_module(&self.path, &self.kind)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ModuleTreeError {
    #[error("module '{path}' has an empty kind")]
    EmptyKind { path: ModulePath },
    #[error("parameter '{parameter}' is not a direct child of module '{module}'")]
    ParameterParentMismatch {
        module: ModulePath,
        parameter: ModulePath,
    },
    #[error("module '{child}' is not a direct child of module '{parent}'")]
    ModuleParentMismatch {
        parent: ModulePath,
        child: ModulePath,
    },
    #[error("duplicate parameter path '{path}' in a module")]
    DuplicateParameterPath { path: ModulePath },
    #[error("duplicate parameter id {id} in a module")]
    DuplicateParameterId { id: ParameterId },
    #[error("duplicate child module path '{path}'")]
    DuplicateModulePath { path: ModulePath },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("module visitor failed: {message}")]
pub struct ModuleVisitError {
    message: String,
}

impl ModuleVisitError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}
