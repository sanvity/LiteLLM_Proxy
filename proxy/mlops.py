import os
import json
import shutil
import time
from typing import Dict, List, Any, Optional

class ModelRegistry:
    def __init__(self, models_dir: str):
        self.models_dir = os.path.abspath(models_dir)
        self.versions_dir = os.path.join(self.models_dir, "versions")
        self.registry_path = os.path.join(self.models_dir, "versions_registry.json")
        self.staging_dir = os.path.join(self.models_dir, "finetuned-deberta")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.versions_dir, exist_ok=True)
        
        self._init_registry()
        
    def _init_registry(self):
        if not os.path.exists(self.registry_path):
            self._save_registry({"active_version": None, "versions": {}})
            
    def _load_registry(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"active_version": None, "versions": {}}
        
    def _save_registry(self, data: Dict[str, Any]):
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ModelRegistry] Error saving registry: {e}")
            
    def get_versions(self) -> List[Dict[str, Any]]:
        registry = self._load_registry()
        versions = registry.get("versions", {})
        active_version = registry.get("active_version")
        
        # Verify paths exist on disk and filter out deleted ones
        valid_versions = {}
        changed = False
        
        for vid, vinfo in list(versions.items()):
            vpath = os.path.join(self.versions_dir, vid)
            if os.path.exists(os.path.join(vpath, "config.json")) or os.path.exists(os.path.join(vpath, "adapter_config.json")):
                vinfo["status"] = "active" if vid == active_version else "inactive"
                valid_versions[vid] = vinfo
            else:
                changed = True
                if vid == active_version:
                    registry["active_version"] = None
                    
        if changed:
            registry["versions"] = valid_versions
            self._save_registry(registry)
            
        # Sort versions by creation date descending
        return sorted(valid_versions.values(), key=lambda x: x.get("created_at", ""), reverse=True)
        
    def save_version(self, name: str, description: str, config: Dict[str, Any]) -> Dict[str, Any]:
        # Verify staging has config.json or adapter_config.json
        if not os.path.exists(os.path.join(self.staging_dir, "config.json")) and not os.path.exists(os.path.join(self.staging_dir, "adapter_config.json")):
            raise ValueError("No fine-tuned model found in staging area to save.")
            
        registry = self._load_registry()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        version_id = f"ver_{timestamp}"
        
        version_path = os.path.join(self.versions_dir, version_id)
        os.makedirs(version_path, exist_ok=True)
        
        # Copy all files from staging_dir to version_path
        for filename in os.listdir(self.staging_dir):
            src_file = os.path.join(self.staging_dir, filename)
            # Skip subdirectories (like checkpoints) and keep only files
            if os.path.isfile(src_file):
                shutil.copy(src_file, os.path.join(version_path, filename))
                
        # Construct metadata
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Check if evaluation report or data exists in staging
        metrics = None
        eval_data_path = os.path.join(self.staging_dir, "evaluation_data.json")
        if os.path.exists(eval_data_path):
            try:
                with open(eval_data_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                    ft_metrics = eval_data.get("finetuned_metrics", {})
                    metrics = ft_metrics.get("macro", {})
            except Exception:
                pass
                
        vinfo = {
            "id": version_id,
            "name": name,
            "description": description,
            "created_at": created_at,
            "epochs": config.get("epochs"),
            "learning_rate": config.get("learning_rate"),
            "batch_size": config.get("batch_size"),
            "dataset_size": config.get("dataset_size"),
            "metrics": metrics,
            "status": "inactive"
        }
        
        registry["versions"][version_id] = vinfo
        self._save_registry(registry)
        return vinfo
        
    def deploy_version(self, version_id: Optional[str]) -> str:
        registry = self._load_registry()
        if version_id is not None:
            if version_id not in registry.get("versions", {}):
                raise ValueError(f"Version {version_id} not found in registry.")
            vpath = os.path.join(self.versions_dir, version_id)
            if not os.path.exists(os.path.join(vpath, "config.json")) and not os.path.exists(os.path.join(vpath, "adapter_config.json")):
                raise ValueError(f"Version folder {version_id} is missing configuration files.")
                
        registry["active_version"] = version_id
        self._save_registry(registry)
        return "Base Model (Default)" if version_id is None else f"Version {version_id}"
        
    def get_active_version(self) -> Optional[str]:
        return self._load_registry().get("active_version")
        
    def delete_version(self, version_id: str):
        registry = self._load_registry()
        if version_id in registry.get("versions", {}):
            # Delete files
            version_path = os.path.join(self.versions_dir, version_id)
            if os.path.exists(version_path):
                shutil.rmtree(version_path, ignore_errors=True)
                
            # Remove from registry
            del registry["versions"][version_id]
            if registry.get("active_version") == version_id:
                registry["active_version"] = None
                
            self._save_registry(registry)
            
    def update_version_metrics(self, version_id: str, metrics: Dict[str, Any]):
        registry = self._load_registry()
        if version_id in registry.get("versions", {}):
            registry["versions"][version_id]["metrics"] = metrics
            self._save_registry(registry)
