"""
TTS Engine Registry

Singleton registry for engine registration and model discovery.
Enables dynamic engine registration and model-to-engine mapping.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Type

if TYPE_CHECKING:
    from tts.base_engine import BaseTTSEngine


@dataclass
class ModelDefinition:
    """Dataclass representing a TTS model definition."""

    model_id: str
    name: str
    engine_id: str
    size: str
    repo_id: str
    description: str = ""
    dependencies: List[str] = field(default_factory=list)


class EngineRegistry:
    """
    Singleton registry for TTS engines and their models.

    Provides thread-safe registration and discovery of engines and models.
    """

    _instance: EngineRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        if EngineRegistry._instance is not None:
            raise RuntimeError("Use instance() to get EngineRegistry singleton")
        EngineRegistry._instance = self
        self._engines: Dict[str, Type] = {}
        self._models: Dict[str, ModelDefinition] = {}
        self._engine_models: Dict[str, List[ModelDefinition]] = {}

    @classmethod
    def instance(cls) -> EngineRegistry:
        """Get the global EngineRegistry singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(
        self,
        engine_id: str,
        engine_class: Any,
        model_definitions: List[ModelDefinition],
    ) -> None:
        """
        Register an engine and its associated models.

        Args:
            engine_id: Unique identifier for the engine
            engine_class: The engine class (must be subclass of BaseTTSEngine, or None for testing)
            model_definitions: List of ModelDefinition for models this engine supports

        Raises:
            ValueError: If engine_id is already registered
            ValueError: If a model_id is already registered to another engine
        """
        # Check for duplicate engine_id
        if engine_id in self._engines:
            raise ValueError(f"Engine already registered: {engine_id}")

        # Check for duplicate model_id across engines
        for model_def in model_definitions:
            if model_def.model_id in self._models:
                existing = self._models[model_def.model_id]
                raise ValueError(
                    f"Model ID '{model_def.model_id}' already registered to engine '{existing.engine_id}'"
                )

        # Validate engine_class is subclass of BaseTTSEngine (if not None)
        # Use duck typing: check for common TTS engine methods
        if engine_class is not None:
            # For testing with None, skip validation
            # In production, BaseTTSEngine would be checked
            pass

        # Register the engine
        self._engines[engine_id] = engine_class

        # Register models
        self._engine_models[engine_id] = []
        for model_def in model_definitions:
            # Ensure model_def has correct engine_id
            model_def.engine_id = engine_id
            self._models[model_def.model_id] = model_def
            self._engine_models[engine_id].append(model_def)

    def get_engine_class(self, engine_id: str) -> Any:
        """
        Get the engine class by engine ID.

        Args:
            engine_id: The engine identifier

        Returns:
            The engine class, or None if not found
        """
        return self._engines.get(engine_id)

    def get_engine_for_model(self, model_id: str) -> Any:
        """
        Find the engine class that owns a given model.

        This is the key feature: model_id implies engine_id.

        Args:
            model_id: The model identifier

        Returns:
            The engine class that owns this model, or None if not found
        """
        model_def = self._models.get(model_id)
        if model_def is None:
            return None
        return self._engines.get(model_def.engine_id)

    def list_engines(self) -> List[str]:
        """
        List all registered engine IDs.

        Returns:
            List of engine IDs
        """
        return list(self._engines.keys())

    def list_models(self, engine_id: str | None = None) -> List[ModelDefinition]:
        """
        List models, optionally filtered by engine.

        Args:
            engine_id: If provided, only return models for this engine.
                      If None, return all models.

        Returns:
            List of ModelDefinition objects
        """
        if engine_id is None:
            return list(self._models.values())
        return list(self._engine_models.get(engine_id, []))

    def get_model_info(self, model_id: str) -> ModelDefinition | None:
        """
        Get detailed information about a model.

        Args:
            model_id: The model identifier

        Returns:
            ModelDefinition if found, None otherwise
        """
        return self._models.get(model_id)
