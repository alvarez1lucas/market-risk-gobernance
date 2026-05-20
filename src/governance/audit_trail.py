"""
src/governance/audit_trail.py — Audit trail inmutable de decisiones del modelo
Requerido por EU AI Act Art. 12 (Record-keeping) y SR 11-7.
Cada evento se hashea con el anterior para detectar manipulaciones.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from loguru import logger


class AuditTrail:
    """
    Log de eventos inmutable basado en cadena de hashes.
    Cada entrada incluye el hash del evento anterior —
    cualquier modificación rompe la cadena y es detectable.
    """

    def __init__(self, log_path: str = "reports/audit_trail.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._get_last_hash()

    def log_event(self, event_type: str, payload: Dict[str, Any],
                  actor: str = "pipeline"):
        """
        Registra un evento en el audit trail.
        
        event_type: tipo de evento (ej: "pipeline_completed", "threshold_changed")
        payload: datos del evento (serializable a JSON)
        actor: quién disparó el evento
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "actor": actor,
            "payload": self._sanitize(payload),
            "previous_hash": self._last_hash,
        }

        # Hash de esta entrada
        entry_str = json.dumps(entry, sort_keys=True, default=str)
        entry["hash"] = hashlib.sha256(entry_str.encode()).hexdigest()
        self._last_hash = entry["hash"]

        # Append — nunca sobreescribir
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        logger.debug(f"Audit trail: [{event_type}] hash={entry['hash'][:8]}...")

    def verify_integrity(self) -> bool:
        """
        Verifica que la cadena de hashes no fue manipulada.
        Retorna True si el trail es íntegro, False si fue alterado.
        """
        if not self.log_path.exists():
            return True

        entries = [json.loads(line) for line in self.log_path.read_text().splitlines() if line]
        if not entries:
            return True

        prev_hash = "GENESIS"
        for entry in entries:
            stored_hash = entry.pop("hash")
            entry_str = json.dumps(entry, sort_keys=True, default=str)
            computed = hashlib.sha256(entry_str.encode()).hexdigest()
            if computed != stored_hash:
                logger.error(f"Audit trail COMPROMETIDO en evento: {entry.get('event_type')}")
                return False
            if entry["previous_hash"] != prev_hash:
                logger.error("Cadena de hashes rota — posible manipulación")
                return False
            prev_hash = stored_hash
            entry["hash"] = stored_hash  # Restaurar

        logger.info(f"Audit trail íntegro — {len(entries)} eventos verificados")
        return True

    def _get_last_hash(self) -> str:
        if not self.log_path.exists():
            return "GENESIS"
        lines = [l for l in self.log_path.read_text().splitlines() if l]
        if not lines:
            return "GENESIS"
        last = json.loads(lines[-1])
        return last.get("hash", "GENESIS")

    def _sanitize(self, payload: Dict) -> Dict:
        """Convierte tipos no serializables a string."""
        result = {}
        for k, v in payload.items():
            try:
                json.dumps(v)
                result[k] = v
            except (TypeError, ValueError):
                result[k] = str(v)
        return result
