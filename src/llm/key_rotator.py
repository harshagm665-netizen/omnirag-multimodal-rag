import os
import itertools
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class KeyRotator:
    def __init__(self):
        load_dotenv()
        self.keys = []
        
        # Primary Key
        primary = os.environ.get("llm_api_key")
        if primary:
            self.keys.append(primary)
            
        # Backup Keys
        backups = os.environ.get("GROQ_BACKUP_KEYS", "")
        if backups:
            for k in backups.split(","):
                k = k.strip()
                if k and k not in self.keys:
                    self.keys.append(k)
                    
        if not self.keys:
            logger.warning("No LLM API keys found in environment variables!")
            
        self._key_iterator = itertools.cycle(self.keys) if self.keys else None

    def get_next_key(self) -> str:
        """Returns the next key in the rotation."""
        if not self._key_iterator:
            raise ValueError("No API keys configured.")
        return next(self._key_iterator)
