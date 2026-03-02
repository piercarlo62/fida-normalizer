"""
TextNormalizer.py - Refactored class for text normalization with preprocessing workflow.
"""
import re
from phonemizer.backend import EspeakBackend
from typing import Tuple, List
import unicodedata

class Normalizer:
    """
    Text normalizer class that preprocesses text to handle specific patterns before NeMo normalization.
    
    Args:
        input_case: Input text capitalization ('cased' or 'lower_cased')
        lang: Language for normalization (default: 'it')
        verbose: Whether to allow output during normalization (default: False)
    """

    def __init__(self, 
                 lang: str = 'it', 
                 tts_mode: bool = True,
                 to_lower: bool = True,
                 phonemize: bool = False):
        self.lang = lang
        self.tts_mode = tts_mode
        self.to_lower = to_lower
        self.phonemize = phonemize

        if phonemize:
            self.phonemizer = EspeakBackend(lang, preserve_punctuation=True, with_stress=False)

        # Language-specific mappings for spoken forms
        self._init_language_mappings()

    def _init_language_mappings(self):
        """Initialize language-specific spoken form mappings."""
        
        # Mapping for spoken form of '.' in European languages
        self.dot_spoken = {
            'it': 'punto',
            'en': 'dot',
            'fr': 'point',
            'de': 'punkt',
            'es': 'punto',
            'pt': 'ponto',
            'nl': 'punt',
            'sv': 'punkt'
        }
        
        # Number to words mapping for different languages
        self._init_number_to_words_mappings()

        # Mapping for spoken form of ',' in European languages
        self.comma_spoken = {
            'it': 'virgola',
            'en': 'comma',
            'fr': 'virgule',
            'de': 'komma',
            'es': 'coma',
            'pt': 'vírgula',
            'nl': 'komma',
            'sv': 'komma'
        }

        # Mapping for spoken form of '@' in European languages
        self.at_spoken = {
            'it': 'chiocciola',
            'en': 'at',
            'fr': 'arobase',
            'de': 'at',
            'es': 'arroba',
            'pt': 'arroba',
            'nl': 'at',
            'sv': 'snabel-a'
        }

        # Mapping for spoken form of '-' and '_' in European languages
        self.hyphen_spoken = {
            'it': 'trattino',
            'en': 'hyphen',
            'fr': 'trait d\'union',
            'de': 'bindestrich',
            'es': 'guion',
            'pt': 'hifen',
            'nl': 'koppelteken',
            'sv': 'bindestreck'
        }

        # Mapping for spoken form of '-' as minus/negative in European languages
        self.minus_spoken = {
            'it': 'meno',
            'en': 'minus',
            'fr': 'moins',
            'de': 'minus',
            'es': 'menos',
            'pt': 'menos',
            'nl': 'min',
            'sv': 'minus'
        }

        self.underscore_spoken = {
            'it': 'trattino basso',
            'en': 'underscore',
            'fr': 'tiret bas',
            'de': 'unterstrich',
            'es': 'guion bajo',
            'pt': 'sublinhado',
            'nl': 'onderstreep',
            'sv': 'understreck'
        }

        # Mapping for spoken form of ':' in European languages (for hours)
        self.colon_spoken = {
            'it': 'e',
            'en': 'colon',
            'fr': 'deux points',
            'de': 'doppelpunkt',
            'es': 'dos puntos',
            'pt': 'dois pontos',
            'nl': 'dubbele punt',
            'sv': 'kolon'
        }

        # Mapping for spoken form of ':' in European languages (for domains)
        self.domain_colon_spoken = {
            'it': 'due punti',
            'en': 'colon',
            'fr': 'deux points',
            'de': 'doppelpunkt',
            'es': 'dos puntos',
            'pt': 'dois pontos',
            'nl': 'dubbele punt',
            'sv': 'kolon'
        }

        # Time unit names in different languages
        self.time_units = {
            'it': {
                'hours': 'ore',
                'minutes': 'minuti', 
                'seconds': 'secondi'
            },
            'en': {
                'hours': 'hours',
                'minutes': 'minutes',
                'seconds': 'seconds'
            },
            'fr': {
                'hours': 'heures',
                'minutes': 'minutes',
                'seconds': 'secondes'
            },
            'de': {
                'hours': 'stunden',
                'minutes': 'minuten',
                'seconds': 'sekunden'
            },
            'es': {
                'hours': 'horas',
                'minutes': 'minutos',
                'seconds': 'segundos'
            },
            'pt': {
                'hours': 'horas',
                'minutes': 'minutos',
                'seconds': 'segundos'
            },
            'nl': {
                'hours': 'uren',
                'minutes': 'minuten',
                'seconds': 'seconden'
            },
            'sv': {
                'hours': 'timmar',
                'minutes': 'minuter',
                'seconds': 'sekunder'
            }
        }

        # Mapping for spoken form of '/' in European languages
        self.slash_spoken = {
            'it': 'slash',
            'en': 'slash',
            'fr': 'slash',
            'de': 'schrägstrich',
            'es': 'barra',
            'pt': 'barra',
            'nl': 'schuine streep',
            'sv': 'snedstreck'
        }

        # Foreign characters mapping to language-specific equivalents
        self.foreign_chars_mapping = {
            'it': {
                # Czech and Slovak characters
                'č': 'c', 'Č': 'C', 'š': 's', 'Š': 'S', 'ž': 'z', 'Ž': 'Z',
                # Polish characters
                'ł': 'l', 'Ł': 'L', 'ń': 'n', 'Ń': 'N', 'ę': 'e', 'Ę': 'E', 'ą': 'a', 'Ą': 'A',
                # German characters
                'ä': 'ae', 'Ä': 'Ae', 'ö': 'oe', 'Ö': 'Oe', 'ü': 'ue', 'Ü': 'Ue', 'ß': 'ss',
                # French characters
                'é': 'e', 'ê': 'e', 'ë': 'e', 'â': 'a', 'î': 'i', 'ï': 'i', 'ô': 'o', 'û': 'u', 'ç': 'c', 'Ç': 'C', 'á': 'a', 'Á': 'A',
                # Spanish characters
                'ñ': 'n', 'Ñ': 'N', 'Á': 'A', 'í': 'i', 'Í': 'I', 'ó': 'o', 'Ó': 'O', 'ú': 'u', 'Ú': 'U',
                # Portuguese characters
                'ã': 'a', 'Ã': 'A', 'õ': 'o', 'Õ': 'O',
                # Scandinavian characters
                'ø': 'o', 'Ø': 'O', 'å': 'a', 'Å': 'A',
                # Other European characters
                'ś': 's', 'Ś': 'S', 'ć': 'c', 'Ć': 'C', 'ź': 'z', 'Ź': 'Z', 'ż': 'z', 'Ż': 'Z'
            },
            'en': {
                # Czech and Slovak characters
                'č': 'c', 'Č': 'C', 'š': 's', 'Š': 'S', 'ž': 'z', 'Ž': 'Z',
                # Polish characters
                'ł': 'l', 'Ł': 'L', 'ń': 'n', 'Ń': 'N', 'ę': 'e', 'Ę': 'E',
                # German characters
                'ä': 'a', 'Ä': 'A', 'ö': 'o', 'Ö': 'O', 'ü': 'u', 'Ü': 'U', 'ß': 'ss',
                # French characters
                'é': 'e', 'ê': 'e', 'ë': 'e', 'â': 'a', 'î': 'i', 'ï': 'i', 'ô': 'o', 'û': 'u', 'ç': 'c', 'Ç': 'C',
                # Spanish characters
                'ñ': 'n', 'Ñ': 'N', 'Á': 'A', 'í': 'i', 'Í': 'I', 'ó': 'o', 'Ó': 'O', 'ú': 'u', 'Ú': 'U',
                # Portuguese characters
                'ã': 'a', 'Ã': 'A', 'õ': 'o', 'Õ': 'O',
                # Scandinavian characters
                'ø': 'o', 'Ø': 'O', 'å': 'a', 'Å': 'A',
                # Other European characters
                'ś': 's', 'Ś': 'S', 'ć': 'c', 'Ć': 'C', 'ź': 'z', 'Ź': 'Z', 'ż': 'z', 'Ż': 'Z'
            },
            'fr': {
                # Czech and Slovak characters
                'č': 'c', 'Č': 'C', 'š': 's', 'Š': 'S', 'ž': 'z', 'Ž': 'Z',
                # Polish characters
                'ł': 'l', 'Ł': 'L', 'ń': 'n', 'Ń': 'N', 'ę': 'e', 'Ę': 'E',
                # German characters (keep German specific mappings)
                'ä': 'a', 'Ä': 'A', 'ö': 'o', 'Ö': 'O', 'ü': 'u', 'Ü': 'U', 'ß': 'ss',
                # French characters (identity mappings)
                'é': 'é', 'ê': 'ê', 'ë': 'ë', 'â': 'â', 'î': 'î', 'ï': 'ï', 'ô': 'ô', 'û': 'û', 'ç': 'ç', 'Ç': 'Ç',
                # Spanish characters
                'ñ': 'n', 'Ñ': 'N', 'Á': 'A', 'í': 'i', 'Í': 'I', 'ó': 'o', 'Ó': 'O', 'ú': 'u', 'Ú': 'U',
                # Portuguese characters
                'ã': 'a', 'Ã': 'A', 'õ': 'o', 'Õ': 'O',
                # Scandinavian characters
                'ø': 'o', 'Ø': 'O', 'å': 'a', 'Å': 'A',
                # Other European characters
                'ś': 's', 'Ś': 'S', 'ć': 'c', 'Ć': 'C', 'ź': 'z', 'Ź': 'Z', 'ż': 'z', 'Ż': 'Z'
            },
            'de': {
                # Czech and Slovak characters
                'č': 'c', 'Č': 'C', 'š': 's', 'Š': 'S', 'ž': 'z', 'Ž': 'Z',
                # Polish characters
                'ł': 'l', 'Ł': 'L', 'ń': 'n', 'Ń': 'N', 'ę': 'e', 'Ę': 'E',
                # German characters (identity mappings for German-specific chars)
                'ä': 'ä', 'Ä': 'Ä', 'ö': 'ö', 'Ö': 'Ö', 'ü': 'ü', 'Ü': 'Ü', 'ß': 'ß',
                # French characters
                'é': 'e', 'ê': 'e', 'ë': 'e', 'â': 'a', 'î': 'i', 'ï': 'i', 'ô': 'o', 'û': 'u', 'ç': 'c', 'Ç': 'C',
                # Spanish characters
                'ñ': 'n', 'Ñ': 'N', 'Á': 'A', 'í': 'i', 'Í': 'I', 'ó': 'o', 'Ó': 'O', 'ú': 'u', 'Ú': 'U',
                # Portuguese characters
                'ã': 'a', 'Ã': 'A', 'õ': 'o', 'Õ': 'O',
                # Scandinavian characters
                'ø': 'o', 'Ø': 'O', 'å': 'a', 'Å': 'A',
                # Other European characters
                'ś': 's', 'Ś': 'S', 'ć': 'c', 'Ć': 'C', 'ź': 'z', 'Ź': 'Z', 'ż': 'z', 'Ż': 'Z'
            },
            'es': {
                # Czech and Slovak characters
                'č': 'c', 'Č': 'C', 'š': 's', 'Š': 'S', 'ž': 'z', 'Ž': 'Z',
                # Polish characters
                'ł': 'l', 'Ł': 'L', 'ń': 'n', 'Ń': 'N', 'ę': 'e', 'Ę': 'E',
                # German characters
                'ä': 'a', 'Ä': 'A', 'ö': 'o', 'Ö': 'O', 'ü': 'u', 'Ü': 'U', 'ß': 'ss',
                # French characters
                'é': 'e', 'ê': 'e', 'ë': 'e', 'â': 'a', 'î': 'i', 'ï': 'i', 'ô': 'o', 'û': 'u', 'ç': 'c', 'Ç': 'C',
                # Spanish characters (identity mappings for Spanish-specific chars)
                'ñ': 'ñ', 'Ñ': 'Ñ', 'Á': 'Á', 'í': 'í', 'Í': 'Í', 'ó': 'ó', 'Ó': 'Ó', 'ú': 'ú', 'Ú': 'Ú',
                # Portuguese characters
                'ã': 'a', 'Ã': 'A', 'õ': 'o', 'Õ': 'O',
                # Scandinavian characters
                'ø': 'o', 'Ø': 'O', 'å': 'a', 'Å': 'A',
                # Other European characters
                'ś': 's', 'Ś': 'S', 'ć': 'c', 'Ć': 'C', 'ź': 'z', 'Ź': 'Z', 'ż': 'z', 'Ż': 'Z'
            },
            'pt': {
                # Czech and Slovak characters
                'č': 'c', 'Č': 'C', 'š': 's', 'Š': 'S', 'ž': 'z', 'Ž': 'Z',
                # Polish characters
                'ł': 'l', 'Ł': 'L', 'ń': 'n', 'Ń': 'N', 'ę': 'e', 'Ę': 'E',
                # German characters
                'ä': 'a', 'Ä': 'A', 'ö': 'o', 'Ö': 'O', 'ü': 'u', 'Ü': 'U', 'ß': 'ss',
                # French characters
                'é': 'e', 'ê': 'e', 'ë': 'e', 'â': 'a', 'î': 'i', 'ï': 'i', 'ô': 'o', 'û': 'u', 'ç': 'c', 'Ç': 'C',
                # Spanish characters
                'ñ': 'n', 'Ñ': 'N', 'Á': 'A', 'í': 'i', 'Í': 'I', 'ó': 'o', 'Ó': 'O', 'ú': 'u', 'Ú': 'U',
                # Portuguese characters (identity mappings for Portuguese-specific chars)
                'ã': 'ã', 'Ã': 'Ã', 'õ': 'õ', 'Õ': 'Õ',
                # Scandinavian characters
                'ø': 'o', 'Ø': 'O', 'å': 'a', 'Å': 'A',
                # Other European characters
                'ś': 's', 'Ś': 'S', 'ć': 'c', 'Ć': 'C', 'ź': 'z', 'Ź': 'Z', 'ż': 'z', 'Ż': 'Z'
            },
            'nl': {
                # Czech and Slovak characters
                'č': 'c', 'Č': 'C', 'š': 's', 'Š': 'S', 'ž': 'z', 'Ž': 'Z',
                # Polish characters
                'ł': 'l', 'Ł': 'L', 'ń': 'n', 'Ń': 'N', 'ę': 'e', 'Ę': 'E',
                # German characters
                'ä': 'a', 'Ä': 'A', 'ö': 'o', 'Ö': 'O', 'ü': 'u', 'Ü': 'U', 'ß': 'ss',
                # French characters
                'é': 'e', 'ê': 'e', 'ë': 'e', 'â': 'a', 'î': 'i', 'ï': 'i', 'ô': 'o', 'û': 'u', 'ç': 'c', 'Ç': 'C',
                # Spanish characters
                'ñ': 'n', 'Ñ': 'N', 'Á': 'A', 'í': 'i', 'Í': 'I', 'ó': 'o', 'Ó': 'O', 'ú': 'u', 'Ú': 'U',
                # Portuguese characters
                'ã': 'a', 'Ã': 'A', 'õ': 'o', 'Õ': 'O',
                # Scandinavian characters
                'ø': 'o', 'Ø': 'O', 'å': 'a', 'Å': 'A',
                # Other European characters
                'ś': 's', 'Ś': 'S', 'ć': 'c', 'Ć': 'C', 'ź': 'z', 'Ź': 'Z', 'ż': 'z', 'Ż': 'Z'
            },
            'sv': {
                # Czech and Slovak characters
                'č': 'c', 'Č': 'C', 'š': 's', 'Š': 'S', 'ž': 'z', 'Ž': 'Z',
                # Polish characters
                'ł': 'l', 'Ł': 'L', 'ń': 'n', 'Ń': 'N', 'ę': 'e', 'Ę': 'E',
                # German characters
                'ä': 'a', 'Ä': 'A', 'ö': 'o', 'Ö': 'O', 'ü': 'u', 'Ü': 'U', 'ß': 'ss',
                # French characters
                'é': 'e', 'ê': 'e', 'ë': 'e', 'â': 'a', 'î': 'i', 'ï': 'i', 'ô': 'o', 'û': 'u', 'ç': 'c', 'Ç': 'C',
                # Spanish characters
                'ñ': 'n', 'Ñ': 'N', 'Á': 'A', 'í': 'i', 'Í': 'I', 'ó': 'o', 'Ó': 'O', 'ú': 'u', 'Ú': 'U',
                # Portuguese characters
                'ã': 'a', 'Ã': 'A', 'õ': 'o', 'Õ': 'O',
                # Scandinavian characters (identity mappings for Swedish-specific chars)
                'ø': 'ø', 'Ø': 'Ø', 'å': 'å', 'Å': 'Å',
                # Other European characters
                'ś': 's', 'Ś': 'S', 'ć': 'c', 'Ć': 'C', 'ź': 'z', 'Ź': 'Z', 'ż': 'z', 'Ż': 'Z'
            }
        }

        # Special characters that should be replaced with spaces
        self.special_chars_to_replace = [
            '«', '»', '#', '^', '*', '|', '\\', '(', ')', '_', '<', '>', '~', '`', '=', '+', '{', '}', '[', ']', '"', ';', ':', '!', '?', '…', '—',','
        ]

        self.tts_special_chars_to_replace = [
            '«', '»', '#', '^', '*', '|', '\\', '(', ')', '_', '<', '>', '~', '`', '=', '+', '{', '}', '[', ']', '"', '—', '…'
        ]

        # Month names in different languages
        self.months = {
            'it': {
                1: 'gennaio', 2: 'febbraio', 3: 'marzo', 4: 'aprile', 5: 'maggio', 6: 'giugno',
                7: 'luglio', 8: 'agosto', 9: 'settembre', 10: 'ottobre', 11: 'novembre', 12: 'dicembre'
            },
            'en': {
                1: 'january', 2: 'february', 3: 'march', 4: 'april', 5: 'may', 6: 'june',
                7: 'july', 8: 'august', 9: 'september', 10: 'october', 11: 'november', 12: 'december'
            },
            'fr': {
                1: 'janvier', 2: 'février', 3: 'mars', 4: 'avril', 5: 'mai', 6: 'juin',
                7: 'juillet', 8: 'août', 9: 'septembre', 10: 'octobre', 11: 'novembre', 12: 'décembre'
            },
            'de': {
                1: 'januar', 2: 'februar', 3: 'märz', 4: 'april', 5: 'mai', 6: 'juni',
                7: 'juli', 8: 'august', 9: 'september', 10: 'oktober', 11: 'november', 12: 'dezember'
            },
            'es': {
                1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril', 5: 'mayo', 6: 'junio',
                7: 'julio', 8: 'agosto', 9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
            },
            'pt': {
                1: 'janeiro', 2: 'fevereiro', 3: 'março', 4: 'abril', 5: 'maio', 6: 'junho',
                7: 'julho', 8: 'agosto', 9: 'setembro', 10: 'outubro', 11: 'novembro', 12: 'dezembro'
            },
            'nl': {
                1: 'januari', 2: 'februari', 3: 'maart', 4: 'april', 5: 'mei', 6: 'juni',
                7: 'juli', 8: 'augustus', 9: 'september', 10: 'oktober', 11: 'november', 12: 'december'
            },
            'sv': {
                1: 'januari', 2: 'februari', 3: 'mars', 4: 'april', 5: 'maj', 6: 'juni',
                7: 'juli', 8: 'augusti', 9: 'september', 10: 'oktober', 11: 'november', 12: 'december'
            }
        }

        # Language-specific number formatting patterns
        self.number_formats = {
            'en': {
                'decimal_separator': '.',
                'thousands_separators': [',', ' '],
                'decimal_spoken': 'point',
                'thousands_spoken': 'thousand'
            },
            'it': {
                'decimal_separator': ',',
                'thousands_separators': ['.', ' '],
                'decimal_spoken': 'virgola',
                'thousands_spoken': 'mila'
            },
            'fr': {
                'decimal_separator': ',',
                'thousands_separators': [' ', '.'],
                'decimal_spoken': 'virgule',
                'thousands_spoken': 'mille'
            },
            'de': {
                'decimal_separator': ',',
                'thousands_separators': ['.', ' '],
                'decimal_spoken': 'komma',
                'thousands_spoken': 'tausend'
            },
            'es': {
                'decimal_separator': ',',
                'thousands_separators': ['.', ' '],
                'decimal_spoken': 'coma',
                'thousands_spoken': 'mil'
            },
            'pt': {
                'decimal_separator': ',',
                'thousands_separators': ['.', ' '],
                'decimal_spoken': 'vírgula',
                'thousands_spoken': 'mil'
            },
            'nl': {
                'decimal_separator': ',',
                'thousands_separators': ['.', ' '],
                'decimal_spoken': 'komma',
                'thousands_spoken': 'duizend'
            },
            'sv': {
                'decimal_separator': ',',
                'thousands_separators': [' ', '.'],
                'decimal_spoken': 'komma',
                'thousands_spoken': 'tusen'
            }
        }

        # Mapping for spoken form of '%' in European languages
        self.percent_spoken = {
            'it': 'percento',
            'en': 'percent',
            'fr': 'pour cent',
            'de': 'prozent',
            'es': 'por ciento',
            'pt': 'por cento',
            'nl': 'procent',
            'sv': 'procent'
        }

        # Mapping for spoken form of '&' in European languages
        self.and_mapping = {
            'it': 'e',
            'en': 'and',
            'fr': 'et',
            'de': 'und',
            'es': 'y',
            'pt': 'e',
            'nl': 'en',
            'sv': 'och'
        }

        # Mapping for spoken form of currency symbols in European languages
        self.currency_mapping = {
            '$': {
                'it': 'dollaro',
                'en': 'dollar',
                'fr': 'dollar',
                'de': 'Dollar',
                'es': 'dólar',
                'pt': 'dólar',
                'nl': 'dollar',
                'sv': 'dollar'
            },
            '€': {
                'it': 'euro',
                'en': 'euro',
                'fr': 'euro',
                'de': 'Euro',
                'es': 'euro',
                'pt': 'euro',
                'nl': 'euro',
                'sv': 'euro'
            },
            '£': {
                'it': 'sterlina',
                'en': 'pound',
                'fr': 'livre',
                'de': 'Pfund',
                'es': 'libra',
                'pt': 'libra',
                'nl': 'pond',
                'sv': 'pund'
            },
            '¥': {
                'it': 'yen',
                'en': 'yen',
                'fr': 'yen',
                'de': 'Yen',
                'es': 'yen',
                'pt': 'iene',
                'nl': 'yen',
                'sv': 'yen'
            }
        }

        # Time-related keywords for context analysis in different languages
        self.time_keywords = {
            'it': {
                'time_indicators': ['alle', 'dalle', 'ore', 'orario', 'orari', 'mattina', 'pomeriggio', 'sera', 'notte', 'mezzogiorno', 'mezzanotte'],
                'time_separator': 'e',
                'decimal_indicators': ['valore', 'costo', 'prezzo', 'quantità', 'numero', 'formula', 'calcolo', 'percento']
            },
            'en': {
                'time_indicators': ['at', 'from', 'to', 'o\'clock', 'morning', 'afternoon', 'evening', 'night', 'noon', 'midnight'],
                'time_separator': 'and',
                'decimal_indicators': ['value', 'cost', 'price', 'amount', 'number', 'formula', 'calculation']
            },
            'fr': {
                'time_indicators': ['à', 'de', 'matin', 'après-midi', 'soir', 'nuit', 'midi', 'minuit'],
                'time_separator': 'et',
                'decimal_indicators': ['valeur', 'coût', 'prix', 'quantité', 'numéro', 'formule', 'calcul']
            },
            'de': {
                'time_indicators': ['um', 'von', 'bis', 'morgens', 'nachmittags', 'abends', 'nachts', 'mittags', 'mitternachts'],
                'time_separator': 'und',
                'decimal_indicators': ['wert', 'kosten', 'preis', 'menge', 'nummer', 'formel', 'berechnung']
            },
            'es': {
                'time_indicators': ['a', 'de', 'mañana', 'tarde', 'noche', 'mediodía', 'medianoche'],
                'time_separator': 'y',
                'decimal_indicators': ['valor', 'costo', 'precio', 'cantidad', 'número', 'fórmula', 'cálculo']
            },
            'pt': {
                'time_indicators': ['às', 'de', 'manhã', 'tarde', 'noite', 'meio-dia', 'meia-noite'],
                'time_separator': 'e',
                'decimal_indicators': ['valor', 'custo', 'preço', 'quantidade', 'número', 'fórmula', 'cálculo']
            },
            'nl': {
                'time_indicators': ['om', 'van', 'ochtend', 'middag', 'avond', 'nacht', 'middag', 'middernacht'],
                'time_separator': 'en',
                'decimal_indicators': ['waarde', 'kosten', 'prijs', 'hoeveelheid', 'nummer', 'formule', 'berekening']
            },
            'sv': {
                'time_indicators': ['kl', 'på', 'morgon', 'eftermiddag', 'kväll', 'natt', 'middag', 'midnatt'],
                'time_separator': 'och',
                'decimal_indicators': ['värde', 'kostnad', 'pris', 'mängd', 'nummer', 'formel', 'beräkning']
            }
        }

        # Unit symbols mappings for different languages
        self.base_units = {
            'it': {
                'm': 'metri', 'kg': 'chilogrammi', 's': 'secondi', 'A': 'ampere', 'K': 'kelvin', 'mol': 'moli', 'cd': 'candele',
                'rad': 'radianti', 'sr': 'steradianti', 'Hz': 'hertz', 'N': 'newton', 'Pa': 'pascal', 'J': 'joule', 'W': 'watt',
                'C': 'coulomb', 'V': 'volt', 'F': 'farad', 'Ω': 'ohm', 'S': 'siemens', 'Wb': 'weber', 'T': 'tesla', 'H': 'henry',
                'lm': 'lumen', 'lx': 'lux', 'Bq': 'becquerel', 'Gy': 'gray', 'Sv': 'sievert', 'kat': 'katal',
                'min': 'minuti', 'h': 'ore', 'd': 'giorni', 'L': 'litri', 'l': 'litri', 'lt': 'litri', 't': 'tonnellate', '°': 'gradi',
                'eV': 'elettronvolt', 'ha': 'ettari'
            },
            'en': {
                'm': 'meters', 'kg': 'kilograms', 's': 'seconds', 'A': 'amperes', 'K': 'kelvin', 'mol': 'moles', 'cd': 'candelas',
                'rad': 'radians', 'sr': 'steradians', 'Hz': 'hertz', 'N': 'newtons', 'Pa': 'pascals', 'J': 'joules', 'W': 'watts',
                'C': 'coulombs', 'V': 'volts', 'F': 'farads', 'Ω': 'ohms', 'S': 'siemens', 'Wb': 'webers', 'T': 'teslas', 'H': 'henrys',
                'lm': 'lumens', 'lx': 'lux', 'Bq': 'becquerels', 'Gy': 'grays', 'Sv': 'sieverts', 'kat': 'katals',
                'min': 'minutes', 'h': 'hours', 'd': 'days', 'L': 'liters', 'l': 'liters', 't': 'tonnes', '°': 'degrees',
                'eV': 'electronvolts', 'ha': 'hectares'
            },
            'fr': {
                'm': 'mètres', 'kg': 'kilogrammes', 's': 'secondes', 'A': 'ampères', 'K': 'kelvin', 'mol': 'moles', 'cd': 'candelas',
                'rad': 'radians', 'sr': 'stéradians', 'Hz': 'hertz', 'N': 'newtons', 'Pa': 'pascals', 'J': 'joules', 'W': 'watts',
                'C': 'coulombs', 'V': 'volts', 'F': 'farads', 'Ω': 'ohms', 'S': 'siemens', 'Wb': 'webers', 'T': 'teslas', 'H': 'henrys',
                'lm': 'lumens', 'lx': 'lux', 'Bq': 'becquerels', 'Gy': 'grays', 'Sv': 'sieverts', 'kat': 'katals',
                'min': 'minutes', 'h': 'heures', 'd': 'jours', 'L': 'litres', 'l': 'litres', 't': 'tonnes', '°': 'degrés',
                'eV': 'électronvolts', 'ha': 'hectares'
            },
            'de': {
                'm': 'Meter', 'kg': 'Kilogramm', 's': 'Sekunden', 'A': 'Ampere', 'K': 'Kelvin', 'mol': 'Mol', 'cd': 'Candela',
                'rad': 'Radiant', 'sr': 'Steradiant', 'Hz': 'Hertz', 'N': 'Newton', 'Pa': 'Pascal', 'J': 'Joule', 'W': 'Watt',
                'C': 'Coulomb', 'V': 'Volt', 'F': 'Farad', 'Ω': 'Ohm', 'S': 'Siemens', 'Wb': 'Weber', 'T': 'Tesla', 'H': 'Henry',
                'lm': 'Lumen', 'lx': 'Lux', 'Bq': 'Becquerel', 'Gy': 'Gray', 'Sv': 'Sievert', 'kat': 'Katal',
                'min': 'Minuten', 'h': 'Stunden', 'd': 'Tage', 'L': 'Liter', 'l': 'Liter', 't': 'Tonnen', '°': 'Grad',
                'eV': 'Elektronvolt', 'ha': 'Hektar'
            },
            'es': {
                'm': 'metros', 'kg': 'kilogramos', 's': 'segundos', 'A': 'amperios', 'K': 'kelvin', 'mol': 'moles', 'cd': 'candelas',
                'rad': 'radianes', 'sr': 'esteradianes', 'Hz': 'hercios', 'N': 'newtons', 'Pa': 'pascales', 'J': 'julios', 'W': 'vatios',
                'C': 'culombios', 'V': 'voltios', 'F': 'faradios', 'Ω': 'ohmios', 'S': 'siemens', 'Wb': 'webers', 'T': 'teslas', 'H': 'henrios',
                'lm': 'lúmenes', 'lx': 'luxes', 'Bq': 'becquereles', 'Gy': 'grays', 'Sv': 'sieverts', 'kat': 'katales',
                'min': 'minutos', 'h': 'horas', 'd': 'días', 'L': 'litros', 'l': 'litros', 't': 'toneladas', '°': 'grados',
                'eV': 'electronvoltios', 'ha': 'hectáreas'
            },
            'pt': {
                'm': 'metros', 'kg': 'quilogramas', 's': 'segundos', 'A': 'amperes', 'K': 'kelvin', 'mol': 'moles', 'cd': 'candelas',
                'rad': 'radianos', 'sr': 'esteradianos', 'Hz': 'hertz', 'N': 'newtons', 'Pa': 'pascais', 'J': 'joules', 'W': 'watts',
                'C': 'coulombs', 'V': 'volts', 'F': 'farads', 'Ω': 'ohms', 'S': 'siemens', 'Wb': 'webers', 'T': 'teslas', 'H': 'henrys',
                'lm': 'lumens', 'lx': 'lux', 'Bq': 'becquerels', 'Gy': 'grays', 'Sv': 'sieverts', 'kat': 'katals',
                'min': 'minutos', 'h': 'horas', 'd': 'dias', 'L': 'litros', 'l': 'litros', 't': 'toneladas', '°': 'graus',
                'eV': 'elétron-volts', 'ha': 'hectares'
            },
            'nl': {
                'm': 'meter', 'kg': 'kilogram', 's': 'seconde', 'A': 'ampere', 'K': 'kelvin', 'mol': 'mol', 'cd': 'candela',
                'rad': 'radiaal', 'sr': 'steradiaal', 'Hz': 'hertz', 'N': 'newton', 'Pa': 'pascal', 'J': 'joule', 'W': 'watt',
                'C': 'coulomb', 'V': 'volt', 'F': 'farad', 'Ω': 'ohm', 'S': 'siemens', 'Wb': 'weber', 'T': 'tesla', 'H': 'henry',
                'lm': 'lumen', 'lx': 'lux', 'Bq': 'becquerel', 'Gy': 'gray', 'Sv': 'sievert', 'kat': 'katal',
                'min': 'minuut', 'h': 'uur', 'd': 'dag', 'L': 'liter', 'l': 'liter', 't': 'ton', '°': 'graad',
                'eV': 'elektronvolt', 'ha': 'hectare'
            },
            'sv': {
                'm': 'meter', 'kg': 'kilogram', 's': 'sekunder', 'A': 'ampere', 'K': 'kelvin', 'mol': 'mol', 'cd': 'candela',
                'rad': 'radian', 'sr': 'steradian', 'Hz': 'hertz', 'N': 'newton', 'Pa': 'pascal', 'J': 'joule', 'W': 'watt',
                'C': 'coulomb', 'V': 'volt', 'F': 'farad', 'Ω': 'ohm', 'S': 'siemens', 'Wb': 'weber', 'T': 'tesla', 'H': 'henry',
                'lm': 'lumen', 'lx': 'lux', 'Bq': 'becquerel', 'Gy': 'gray', 'Sv': 'sievert', 'kat': 'katal',
                'min': 'minuter', 'h': 'timmar', 'd': 'dagar', 'L': 'liter', 'l': 'liter', 't': 'ton', '°': 'grader',
                'eV': 'elektronvolt', 'ha': 'hektar'
            }
        }

        # Unit prefixes mappings for different languages
        self.unit_prefixes = {
            'it': {'m': 'milli', 'µ': 'micro', 'n': 'nano', 'k': 'chilo', 'M': 'mega', 'G': 'giga', 'T': 'tera'},
            'en': {'m': 'milli', 'µ': 'micro', 'n': 'nano', 'k': 'kilo', 'M': 'mega', 'G': 'giga', 'T': 'tera'},
            'fr': {'m': 'milli', 'µ': 'micro', 'n': 'nano', 'k': 'kilo', 'M': 'méga', 'G': 'giga', 'T': 'téra'},
            'de': {'m': 'milli', 'µ': 'mikro', 'n': 'nano', 'k': 'kilo', 'M': 'mega', 'G': 'giga', 'T': 'tera'},
            'es': {'m': 'mili', 'µ': 'micro', 'n': 'nano', 'k': 'kilo', 'M': 'mega', 'G': 'giga', 'T': 'tera'},
            'pt': {'m': 'mili', 'µ': 'micro', 'n': 'nano', 'k': 'quilo', 'M': 'mega', 'G': 'giga', 'T': 'tera'},
            'nl': {'m': 'milli', 'µ': 'micro', 'n': 'nano', 'k': 'kilo', 'M': 'mega', 'G': 'giga', 'T': 'tera'},
            'sv': {'m': 'milli', 'µ': 'mikro', 'n': 'nano', 'k': 'kilo', 'M': 'mega', 'G': 'giga', 'T': 'tera'}
        }

        # Compound unit symbols mappings for different languages
        self.compound_units = {
            'it': {
                'm/s': 'metri al secondo', 'm/s²': 'metri al secondo quadrato', 'm²': 'metri quadrati', 'm³': 'metri cubi',
                'kg/m³': 'chilogrammi per metro cubo', 'N/m²': 'newton per metro quadrato', 'J/m³': 'joule per metro cubo',
                'km/h': 'chilometri all\'ora'
            },
            'en': {
                'm/s': 'meters per second', 'm/s²': 'meters per second squared', 'm²': 'square meters', 'm³': 'cubic meters',
                'kg/m³': 'kilograms per cubic meter', 'N/m²': 'newtons per square meter', 'J/m³': 'joules per cubic meter',
                'km/h': 'kilometers per hour'
            },
            'fr': {
                'm/s': 'mètres par seconde', 'm/s²': 'mètres par seconde carrée', 'm²': 'mètres carrés', 'm³': 'mètres cubes',
                'kg/m³': 'kilogrammes par mètre cube', 'N/m²': 'newtons par mètre carré', 'J/m³': 'joules par mètre cube',
                'km/h': 'kilomètres par heure'
            },
            'de': {
                'm/s': 'Meter pro Sekunde', 'm/s²': 'Meter pro Sekunde quadrat', 'm²': 'Quadratmeter', 'm³': 'Kubikmeter',
                'kg/m³': 'Kilogramm pro Kubikmeter', 'N/m²': 'Newton pro Quadratmeter', 'J/m³': 'Joule pro Kubikmeter',
                'km/h': 'Kilometer pro Stunde'
            },
            'es': {
                'm/s': 'metros por segundo', 'm/s²': 'metros por segundo cuadrado', 'm²': 'metros cuadrados', 'm³': 'metros cúbicos',
                'kg/m³': 'kilogramos por metro cúbico', 'N/m²': 'newtons por metro cuadrado', 'J/m³': 'julios por metro cúbico',
                'km/h': 'kilómetros por hora'
            },
            'pt': {
                'm/s': 'metros por segundo', 'm/s²': 'metros por segundo quadrado', 'm²': 'metros quadrados', 'm³': 'metros cúbicos',
                'kg/m³': 'quilogramas por metro cúbico', 'N/m²': 'newtons por metro quadrado', 'J/m³': 'joules por metro cúbico',
                'km/h': 'quilómetros por hora'
            },
            'nl': {
                'm/s': 'meter per seconde', 'm/s²': 'meter per seconde kwadraat', 'm²': 'vierkante meter', 'm³': 'kubieke meter',
                'kg/m³': 'kilogram per kubieke meter', 'N/m²': 'newton per vierkante meter', 'J/m³': 'joule per kubieke meter',
                'km/h': 'kilometer per uur'
            },
            'sv': {
                'm/s': 'meter per sekund', 'm/s²': 'meter per sekund kvadrat', 'm²': 'kvadratmeter', 'm³': 'kubikmeter',
                'kg/m³': 'kilogram per kubikmeter', 'N/m²': 'newton per kvadratmeter', 'J/m³': 'joule per kubikmeter',
                'km/h': 'kilometer per timme'
            }
        }

    def _init_number_to_words_mappings(self):
        """Initialize number to words mappings for different languages."""
        
        # Italian numbers
        self.numbers_it = {
            0: 'zero', 1: 'uno', 2: 'due', 3: 'tre', 4: 'quattro', 5: 'cinque',
            6: 'sei', 7: 'sette', 8: 'otto', 9: 'nove', 10: 'dieci',
            11: 'undici', 12: 'dodici', 13: 'tredici', 14: 'quattordici', 15: 'quindici',
            16: 'sedici', 17: 'diciassette', 18: 'diciotto', 19: 'diciannove', 20: 'venti',
            30: 'trenta', 40: 'quaranta', 50: 'cinquanta', 60: 'sessanta', 70: 'settanta',
            80: 'ottanta', 90: 'novanta', 100: 'cento', 1000: 'mila', 1000000: 'milioni'
        }
        
        # English numbers
        self.numbers_en = {
            0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
            6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
            11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen',
            16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty',
            30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy',
            80: 'eighty', 90: 'ninety', 100: 'hundred', 1000: 'thousand', 1000000: 'million'
        }
        
        # French numbers
        self.numbers_fr = {
            0: 'zéro', 1: 'un', 2: 'deux', 3: 'trois', 4: 'quatre', 5: 'cinq',
            6: 'six', 7: 'sept', 8: 'huit', 9: 'neuf', 10: 'dix',
            11: 'onze', 12: 'douze', 13: 'treize', 14: 'quatorze', 15: 'quinze',
            16: 'seize', 17: 'dix-sept', 18: 'dix-huit', 19: 'dix-neuf', 20: 'vingt',
            30: 'trente', 40: 'quarante', 50: 'cinquante', 60: 'soixante', 70: 'soixante-dix',
            80: 'quatre-vingts', 90: 'quatre-vingt-dix', 100: 'cent', 1000: 'mille', 1000000: 'million'
        }
        
        # German numbers
        self.numbers_de = {
            0: 'null', 1: 'eins', 2: 'zwei', 3: 'drei', 4: 'vier', 5: 'fünf',
            6: 'sechs', 7: 'sieben', 8: 'acht', 9: 'neun', 10: 'zehn',
            11: 'elf', 12: 'zwölf', 13: 'dreizehn', 14: 'vierzehn', 15: 'fünfzehn',
            16: 'sechzehn', 17: 'siebzehn', 18: 'achtzehn', 19: 'neunzehn', 20: 'zwanzig',
            30: 'dreißig', 40: 'vierzig', 50: 'fünfzig', 60: 'sechzig', 70: 'siebzig',
            80: 'achtzig', 90: 'neunzig', 100: 'hundert', 1000: 'tausend', 1000000: 'million'
        }
        
        # Spanish numbers
        self.numbers_es = {
            0: 'cero', 1: 'uno', 2: 'dos', 3: 'tres', 4: 'cuatro', 5: 'cinco',
            6: 'seis', 7: 'siete', 8: 'ocho', 9: 'nueve', 10: 'diez',
            11: 'once', 12: 'doce', 13: 'trece', 14: 'catorce', 15: 'quince',
            16: 'dieciséis', 17: 'diecisiete', 18: 'dieciocho', 19: 'diecinueve', 20: 'veinte',
            30: 'treinta', 40: 'cuarenta', 50: 'cincuenta', 60: 'sesenta', 70: 'setenta',
            80: 'ochenta', 90: 'noventa', 100: 'cien', 1000: 'mil', 1000000: 'millón'
        }
        
        # Portuguese numbers
        self.numbers_pt = {
            0: 'zero', 1: 'um', 2: 'dois', 3: 'três', 4: 'quatro', 5: 'cinco',
            6: 'seis', 7: 'sete', 8: 'oito', 9: 'nove', 10: 'dez',
            11: 'onze', 12: 'doze', 13: 'treze', 14: 'catorze', 15: 'quinze',
            16: 'dezesseis', 17: 'dezessete', 18: 'dezoito', 19: 'dezenove', 20: 'vinte',
            30: 'trinta', 40: 'quarenta', 50: 'cinquenta', 60: 'sessenta', 70: 'setenta',
            80: 'oitenta', 90: 'noventa', 100: 'cem', 1000: 'mil', 1000000: 'milhão'
        }
        
        # Dutch numbers
        self.numbers_nl = {
            0: 'nul', 1: 'een', 2: 'twee', 3: 'drie', 4: 'vier', 5: 'vijf',
            6: 'zes', 7: 'zeven', 8: 'acht', 9: 'negen', 10: 'tien',
            11: 'elf', 12: 'twaalf', 13: 'dertien', 14: 'veertien', 15: 'vijftien',
            16: 'zestien', 17: 'zeventien', 18: 'achttien', 19: 'negentien', 20: 'twintig',
            30: 'dertig', 40: 'veertig', 50: 'vijftig', 60: 'zestig', 70: 'zeventig',
            80: 'tachtig', 90: 'negentig', 100: 'honderd', 1000: 'duizend', 1000000: 'miljoen'
        }
        
        # Swedish numbers
        self.numbers_sv = {
            0: 'noll', 1: 'ett', 2: 'två', 3: 'tre', 4: 'fyra', 5: 'fem',
            6: 'sex', 7: 'sju', 8: 'åtta', 9: 'nio', 10: 'tio',
            11: 'elva', 12: 'tolv', 13: 'tretton', 14: 'fjorton', 15: 'femton',
            16: 'sexton', 17: 'sjutton', 18: 'arton', 19: 'nitton', 20: 'tjugo',
            30: 'trettio', 40: 'fyrtio', 50: 'femtio', 60: 'sextio', 70: 'sjuttio',
            80: 'åttio', 90: 'nittio', 100: 'hundra', 1000: 'tusen', 1000000: 'miljon'
        }
        
        # Add billion mappings to all languages
        self.numbers_it[1000000000] = 'miliardi'
        self.numbers_en[1000000000] = 'billion'
        self.numbers_fr[1000000000] = 'milliard'
        self.numbers_de[1000000000] = 'milliarde'
        self.numbers_es[1000000000] = 'mil millones'
        self.numbers_pt[1000000000] = 'bilhão'
        self.numbers_nl[1000000000] = 'miljard'
        self.numbers_sv[1000000000] = 'miljard'
        
        # Map language codes to number dictionaries
        self.number_mappings = {
            'it': self.numbers_it,
            'en': self.numbers_en,
            'fr': self.numbers_fr,
            'de': self.numbers_de,
            'es': self.numbers_es,
            'pt': self.numbers_pt,
            'nl': self.numbers_nl,
            'sv': self.numbers_sv
        }

    def _number_to_words(self, number: int) -> str:
        """Convert a number to its spoken form in the configured language.
        
        Args:
            number: The number to convert
            
        Returns:
            The spoken form of the number
        """
        # Get the number mapping for the current language
        numbers_map = self.number_mappings.get(self.lang, self.numbers_en)
        
        # Handle zero
        if number == 0:
            return numbers_map.get(0, 'zero')
        
        # Handle very large numbers (billions and above)
        if number >= 1000000000:
            billions = number // 1000000000
            remainder = number % 1000000000
            
            billions_word = self._number_to_words(billions)
            
            # Handle singular/plural forms for different languages
            if self.lang == 'it':
                # Italian singular/plural distinction
                if billions == 1:
                    billions_word = 'un'  # Use "un" not "uno" for masculine nouns
                    billions_label = 'miliardo'
                else:
                    billions_word = self._number_to_words(billions)
                    billions_label = 'miliardi'
            else:
                billions_word = self._number_to_words(billions)
                billions_label = numbers_map.get(1000000000, 'billion')
            
            if remainder == 0:
                return f"{billions_word} {billions_label}"
            else:
                remainder_word = self._number_to_words(remainder)
                return f"{billions_word} {billions_label} {remainder_word}"
        
        # Handle millions
        if number >= 1000000:
            millions = number // 1000000
            remainder = number % 1000000
            
            millions_word = self._number_to_words(millions)
            
            # Handle singular/plural forms for different languages
            if self.lang == 'it':
                # Italian singular/plural distinction
                if millions == 1:
                    millions_word = 'un'  # Use "un" not "uno" for masculine nouns
                    millions_label = 'milione'
                else:
                    millions_word = self._number_to_words(millions)
                    millions_label = 'milioni'
            else:
                millions_word = self._number_to_words(millions)
                millions_label = numbers_map.get(1000000, 'million')
            
            if remainder == 0:
                return f"{millions_word} {millions_label}"
            else:
                remainder_word = self._number_to_words(remainder)
                return f"{millions_word} {millions_label} {remainder_word}"
        
        # Handle thousands
        if number >= 1000:
            thousands = number // 1000
            remainder = number % 1000
            
            # Handle thousands with different rules for different languages
            if thousands == 1:
                # "one thousand" vs just "thousand"
                if self.lang in ['en', 'de', 'nl']:
                    thousands_word = numbers_map.get(1, 'one')
                elif self.lang == 'it':
                    # Italian uses "mille" for exactly 1000
                    thousands_word = 'mille'
                else:
                    thousands_word = ''
            else:
                # Special handling for Italian thousands
                if self.lang == 'it':
                    # Italian forms compound thousands: duemila, tremila, etc.
                    if remainder == 0:
                        # Exact thousands in Italian are compound
                        if thousands == 2:
                            thousands_word = 'duemila'
                        elif thousands == 3:
                            thousands_word = 'tremila'
                        elif thousands == 4:
                            thousands_word = 'quattromila'
                        elif thousands == 5:
                            thousands_word = 'cinquemila'
                        elif thousands == 6:
                            thousands_word = 'seimila'
                        elif thousands == 7:
                            thousands_word = 'settemila'
                        elif thousands == 8:
                            thousands_word = 'ottomila'
                        elif thousands == 9:
                            thousands_word = 'novemila'
                        else:
                            # For larger thousands, use separate form without extra "mila"
                            thousands_word = self._number_to_words(thousands)
                    else:
                        # Non-exact thousands - use separate form without extra "mila"
                        thousands_word = self._number_to_words(thousands)
                else:
                    thousands_word = self._number_to_words(thousands)
            
            thousands_label = numbers_map.get(1000, 'thousand')
            
            if remainder == 0:
                if thousands_word:
                    if self.lang == 'it' and thousands == 1:
                        return f"{thousands_word}"  # Only "mille", no "mila"
                    else:
                        return f"{thousands_word} {thousands_label}"
                else:
                    return thousands_label
            else:
                remainder_word = self._number_to_words(remainder)
                if thousands_word:
                    if self.lang == 'it' and thousands == 1:
                        return f"{thousands_word} {remainder_word}"  # Only "mille", no "mila"
                    else:
                        return f"{thousands_word} {thousands_label} {remainder_word}"
                else:
                    return f"{thousands_label} {remainder_word}"
        
        # Handle hundreds
        if number >= 100:
            hundreds_digit = number // 100
            remainder = number % 100
            
            if hundreds_digit == 1:
                # For English, "one hundred" not just "hundred"
                if self.lang == 'en':
                    hundreds_word = f"{numbers_map.get(1, 'one')} {numbers_map.get(100, 'hundred')}"
                else:
                    hundreds_word = numbers_map.get(100, 'hundred')
            else:
                # Special compound handling for Italian, French, Spanish, Portuguese
                if self.lang == 'it':
                    # Italian forms compound hundreds: duecento, trecento, etc.
                    # Use compound forms for all hundreds (2-9), even with remainders
                    if hundreds_digit == 2:
                        hundreds_word = 'duecento'
                    elif hundreds_digit == 3:
                        hundreds_word = 'trecento'
                    elif hundreds_digit == 4:
                        hundreds_word = 'quattrocento'
                    elif hundreds_digit == 5:
                        hundreds_word = 'cinquecento'
                    elif hundreds_digit == 6:
                        hundreds_word = 'seicento'
                    elif hundreds_digit == 7:
                        hundreds_word = 'settecento'
                    elif hundreds_digit == 8:
                        hundreds_word = 'ottocento'
                    elif hundreds_digit == 9:
                        hundreds_word = 'novecento'
                    else:
                        # Use separate form for 100
                        hundreds_digit_word = numbers_map.get(hundreds_digit, str(hundreds_digit))
                        hundreds_word = f"{hundreds_digit_word} {numbers_map.get(100, 'hundred')}"
                elif self.lang == 'fr':
                    # French forms compound hundreds: deux cents (when exact), deux cent (with remainder)
                    if remainder == 0 and hundreds_digit > 1:
                        # French exact hundreds use "cents" (plural)
                        hundreds_digit_word = numbers_map.get(hundreds_digit, str(hundreds_digit))
                        hundreds_word = f"{hundreds_digit_word} cents"
                    else:
                        # French non-exact hundreds use "cent" (singular)
                        hundreds_digit_word = numbers_map.get(hundreds_digit, str(hundreds_digit))
                        hundreds_word = f"{hundreds_digit_word} cent"
                elif self.lang == 'es':
                    # Spanish forms compound hundreds: doscientos, trescientos, etc.
                    # Always use compound form for exact hundreds
                    if hundreds_digit == 2:
                        hundreds_word = 'doscientos'
                    elif hundreds_digit == 3:
                        hundreds_word = 'trescientos'
                    elif hundreds_digit == 4:
                        hundreds_word = 'cuatrocientos'
                    elif hundreds_digit == 5:
                        hundreds_word = 'quinientos'
                    elif hundreds_digit == 6:
                        hundreds_word = 'seiscientos'
                    elif hundreds_digit == 7:
                        hundreds_word = 'setecientos'
                    elif hundreds_digit == 8:
                        hundreds_word = 'ochocientos'
                    elif hundreds_digit == 9:
                        hundreds_word = 'novecientos'
                    else:
                        hundreds_digit_word = numbers_map.get(hundreds_digit, str(hundreds_digit))
                        hundreds_word = f"{hundreds_digit_word} {numbers_map.get(100, 'hundred')}"
                elif self.lang == 'pt':
                    # Portuguese forms compound hundreds: duzentos, trezentos, etc.
                    # Always use compound form for exact hundreds
                    if hundreds_digit == 2:
                        hundreds_word = 'duzentos'
                    elif hundreds_digit == 3:
                        hundreds_word = 'trezentos'
                    elif hundreds_digit == 4:
                        hundreds_word = 'quatrocentos'
                    elif hundreds_digit == 5:
                        hundreds_word = 'quinhentos'
                    elif hundreds_digit == 6:
                        hundreds_word = 'seiscentos'
                    elif hundreds_digit == 7:
                        hundreds_word = 'setecentos'
                    elif hundreds_digit == 8:
                        hundreds_word = 'oitocentos'
                    elif hundreds_digit == 9:
                        hundreds_word = 'novecentos'
                    else:
                        hundreds_digit_word = numbers_map.get(hundreds_digit, str(hundreds_digit))
                        hundreds_word = f"{hundreds_digit_word} {numbers_map.get(100, 'hundred')}"
                else:
                    hundreds_digit_word = numbers_map.get(hundreds_digit, str(hundreds_digit))
                    hundreds_word = f"{hundreds_digit_word} {numbers_map.get(100, 'hundred')}"
            
            if remainder == 0:
                return hundreds_word
            else:
                remainder_word = self._number_to_words(remainder)
                return f"{hundreds_word} {remainder_word}"
        
        # Handle numbers up to 20 directly
        if number <= 20:
            return numbers_map.get(number, str(number))
        
        # Handle tens (21-99)
        if number >= 21:
            tens = (number // 10) * 10
            units = number % 10
            
            if units == 0:
                return numbers_map.get(tens, str(number))
            else:
                # Special compound handling for different languages
                if self.lang == 'fr':
                    # French compound tens: 70=soixante-dix, 80=quatre-vingts, 90=quatre-vingt-dix
                    if tens == 70:
                        # French 70s: 71-79 = 60 + 11-19
                        if units == 5:  # 75
                            return f"soixante quinze"
                        elif units == 7:  # 77  
                            return f"soixante dix-sept"
                        elif units == 8:  # 78
                            return f"soixante dix-huit"
                        elif units == 9:  # 79
                            return f"soixante dix-neuf"
                        else:
                            teens_map = {1: 'onze', 2: 'douze', 3: 'treize', 4: 'quatorze'}
                            if units <= 4:
                                return f"soixante {teens_map[units]}"
                            else:
                                return f"soixante {numbers_map.get(units, str(units))}"
                    elif tens == 90:
                        return f"quatre-vingt {numbers_map.get(units, str(units))}"
                    else:
                        tens_word = numbers_map.get(tens, str(tens))
                        units_word = numbers_map.get(units, str(units))
                        return f"{tens_word} {units_word}"
                else:
                    tens_word = numbers_map.get(tens, str(tens))
                    units_word = numbers_map.get(units, str(units))
                    return f"{tens_word} {units_word}"
        
        # Fallback to string representation
        return str(number)
    
    def _replace_foreign_chars(self, text: str) -> str:
        """
        Replace foreign characters with language-specific equivalents.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with foreign characters replaced by language-specific equivalents
        """
        # Get the foreign character mapping for the current language
        lang_mapping = self.foreign_chars_mapping.get(self.lang, {})
        
        # If no mapping exists for the language, return text unchanged
        if not lang_mapping:
            return text
            
        transformed = text
        for foreign_char, native_char in lang_mapping.items():
            transformed = transformed.replace(foreign_char, native_char)
        
        return transformed

    def _replace_special_chars_with_spaces(self, text: str) -> str:
        """
        Replace special characters with spaces.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with special characters replaced by spaces
        """
        transformed = text
        
        # Replace other special characters with spaces
        if not self.tts_mode:
            for special_char in self.special_chars_to_replace:
                transformed = transformed.replace(special_char, ' ')
        else:
            for special_char in self.tts_special_chars_to_replace:
                transformed = transformed.replace(special_char, ' ')
        
        # Handle hyphens specially - replace with space only if clearly not part of negative numbers
        # This regex matches hyphens that should be replaced with spaces:
        # 1. Between two word characters (letters, digits, underscore) - like "well-known"
        # 2. At the end of the string
        # 3. At the beginning followed by letters (not digits) - like "-test" -> " test"
        # Note: We avoid replacing hyphens that might be part of negative numbers to let number processing handle them
        hyphen_pattern = r'(?<=[\w])-(?=[\w])|-(?=$)|^-(?=[a-zA-Z])'
        transformed = re.sub(hyphen_pattern, ' ', transformed)
        
        return transformed

    def _remove_double_spaces(self, text: str) -> str:
        """Remove double spaces from text."""
        return re.sub(r'\s+', ' ', text).strip()

    def _normalize_leading_zeros_in_decimals(self, text: str) -> str:
        """
        Normalize leading zeros in decimal parts of numbers.
        For example: '1,01' becomes '1, zero 1' so NeMo can process it correctly.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with leading zeros in decimals normalized
        """
        # Pattern to match numbers with leading zeros in decimal parts
        # This matches patterns like: 1,01 or 2,007 or 3,0001
        decimal_pattern = r'(\d+),(\d+)'

        transformed = text
        matches = list(re.finditer(decimal_pattern, transformed))
        for match in reversed(matches):
            integer_part = match.group(1)
            decimal_part = match.group(2)

            # Check if the decimal part has leading zeros
            if len(decimal_part) > 1 and decimal_part[0] == '0':
                # Split the decimal part into leading zeros and significant digits
                significant_part = decimal_part.lstrip('0')
                if significant_part == '':  # All zeros
                    # For something like 1,000 -> '1, zero'
                    replacement = f'{integer_part}, zero'
                else:
                    # For something like 1,01 -> '1, zero 1'
                    replacement = f'{integer_part}, zero {significant_part}'
            else:
                # No leading zeros, return as is
                replacement = match.group(0)

            transformed = transformed[:match.start()] + replacement + transformed[match.end():]
        
        return transformed

    def _transform_currency_symbols(self, text: str) -> str:
        """
        Transform currency symbols to their language-specific spoken forms.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with currency symbols replaced by language-specific spoken forms
        """
        transformed = text
        for symbol, lang_mapping in self.currency_mapping.items():
            spoken = lang_mapping.get(self.lang, symbol)
            transformed = re.sub(re.escape(symbol), f' {spoken} ', transformed)
        
        return transformed

    def _transform_email(self, email: str) -> str:
        """
        Transform email according to specified rules.
        
        Args:
            email: Email string to transform
            
        Returns:
            Transformed email string
        """
        # Get language-specific mappings
        dot_spoken = self.dot_spoken.get(self.lang, 'punto')
        at_spoken = self.at_spoken.get(self.lang, 'chiocciola')
        hyphen_spoken = self.hyphen_spoken.get(self.lang, 'trattino')
        underscore_spoken = self.underscore_spoken.get(self.lang, 'trattino basso')
        
        # Replace '.' with corresponding language spoken form padded with blank space
        transformed = re.sub(r'\.', f' {dot_spoken} ', email)
        
        # Replace '@' with corresponding language spoken form padded by blank space
        transformed = re.sub(r'@', f' {at_spoken} ', transformed)
        
        # Replace '-' or '_' with corresponding language spoken form padded by blank space
        matches = list(re.finditer(r'[-_]', transformed))
        for match in reversed(matches):
            char = match.group()
            replacement = f' {hyphen_spoken} ' if char == '-' else f' {underscore_spoken} '
            transformed = transformed[:match.start()] + replacement + transformed[match.end():]
        
        # Check if in the resulted words there is number in digit way and add padding
        transformed = re.sub(r'(\d+)', r' \1 ', transformed)
        
        # Remove double spaces
        transformed = self._remove_double_spaces(transformed)
        
        return transformed

    def _transform_number_with_digits(self, number: str, context: str = None) -> str:
        """
        Transform number with digits according to language-specific rules.
        Handles decimal and thousands separators based on the language.
        Removes thousands separators and replaces decimal separator with spoken form.
        Also handles negative numbers by replacing minus sign with language-specific spoken form.
        
        Args:
            number: Number string to transform (can include negative sign)
            context: Context hint ('time', 'decimal', or None)
            
        Returns:
            Transformed number string
        """
        # Get language-specific number format settings
        number_format = self.number_formats.get(self.lang, self.number_formats['en'])
        decimal_sep = number_format['decimal_separator']
        decimal_spoken = number_format['decimal_spoken']
        thousands_seps = number_format['thousands_separators']
        
        # Get language-specific mapping for minus
        minus_spoken = self.minus_spoken.get(self.lang, 'meno')
        
        # Handle negative numbers
        is_negative = number.startswith('-')
        if is_negative:
            # Remove the minus sign for processing, we'll add the spoken form back later
            number = number[1:]
        
        # Handle both dot and comma as potential separators for flexibility
        # This allows handling international formats (e.g., "11.30" for Italian times)
        potential_separators = [decimal_sep, '.' if decimal_sep != '.' else ',', ',' if decimal_sep != ',' else '.']
        
        # Remove thousands separators first (they appear before 3 digits)
        # But be smart about it - don't remove separators that look like decimals
        transformed = number
        for sep in thousands_seps:
            if sep != decimal_sep:  # Don't remove the primary decimal separator
                # For potential decimal separators, check if they look like decimals
                # A separator is likely a decimal if:
                # 1. It appears only once in the number, OR
                # 2. The part after it has more than 3 digits (suggesting it's not thousands)
                import re
                sep_positions = [m.start() for m in re.finditer(re.escape(sep), transformed)]
                
                # If separator appears only once and the part after it has >3 digits, skip removal
                if len(sep_positions) == 1:
                    sep_pos = sep_positions[0]
                    part_after = transformed[sep_pos + 1:]
                    if part_after and len(part_after) > 3:
                        continue  # Skip removal - likely a decimal separator
                
                # Remove thousands separators using negative lookahead
                transformed = re.sub(rf'\{sep}(?=\d{{3}})', '', transformed)
        
        # If context suggests this is a time, use time separator instead of decimal
        if context == 'time':
            time_keywords = self.time_keywords.get(self.lang, self.time_keywords['it'])
            time_separator = time_keywords['time_separator']
            
            # Replace any potential decimal separator with time separator
            for sep in potential_separators:
                if sep in transformed:
                    transformed = re.sub(rf'\{sep}', f' {time_separator} ', transformed)
                    break  # Only replace the first separator found
        else:
            # Use standard decimal separator (for 'decimal' context or None)
            for sep in potential_separators:
                if sep in transformed:
                    transformed = re.sub(rf'\{sep}', f' {decimal_spoken} ', transformed)
                    break  # Only replace the first separator found
        
        # Add minus spoken form if the original number was negative
        if is_negative:
            transformed = f'{minus_spoken} {transformed}'
        
        # Clean up any double spaces that may have been introduced
        transformed = self._remove_double_spaces(transformed)

        # Remove spaces before punctuation marks
        transformed = re.sub(r'\s+([.!?;:])', r'\1', transformed)

        return transformed

    def _transform_domain(self, domain: str) -> str:
        """
        Transform domain according to specified rules.
        
        Args:
            domain: Domain string to transform
            
        Returns:
            Transformed domain string
        """
        # Get language-specific mappings
        domain_colon_spoken = self.domain_colon_spoken.get(self.lang, 'due punti')
        slash_spoken = self.slash_spoken.get(self.lang, 'slash')
        dot_spoken = self.dot_spoken.get(self.lang, 'punto')
        
        # Replace http, https, www with whitespace between each char and padding
        protocol_pattern = r'\b(https?|www)\b'
        matches = list(re.finditer(protocol_pattern, domain))
        for match in reversed(matches):
            replacement = ' ' + ' '.join(match.group()) + ' '
            domain = domain[:match.start()] + replacement + domain[match.end():]
        transformed = domain
        
        # Replace ':' with relative language spoken form for domains
        transformed = re.sub(r':', f' {domain_colon_spoken} ', transformed)
        
        # Replace '/' char with relative language spoken form wrapped by blank space
        transformed = re.sub(r'/', f' {slash_spoken} ', transformed)
        
        # Replace '.' char with relative language spoken form wrapped by blank space
        transformed = re.sub(r'\.', f' {dot_spoken} ', transformed)
        
        # Check and remove double blank spaces
        transformed = self._remove_double_spaces(transformed)
        
        return transformed

    def _transform_hours(self, hours: str) -> str:
        """
        Transform hours according to specified rules.
        Replace ":" with language-appropriate spoken form and separators.
        
        Args:
            hours: Hours string to transform (e.g., "2:15:45" or "20:30:00")
            
        Returns:
            Transformed hours string
        """
        # Split the time components
        parts = hours.split(':')
        
        # Get language-specific time keywords for proper separators
        time_keywords = self.time_keywords.get(self.lang, self.time_keywords['it'])
        time_separator = time_keywords['time_separator']
        
        if len(parts) == 2:  # HH:MM format
            hour, minute = parts
            # Format: "hour [separator] minute"
            return f"{hour} {time_separator} {minute}"
            
        elif len(parts) == 3:  # HH:MM:SS format
            hour, minute, second = parts
            
            # Get language-specific time units
            time_units = self.time_units.get(self.lang, self.time_units['it'])
            seconds_unit = time_units['seconds']
            
            # Format: "hour [separator] minute [separator] second seconds_unit"
            return f"{hour} {time_separator} {minute} {time_separator} {second} {seconds_unit}"
        else:
            # Fallback for unexpected formats
            return hours

    def _transform_date(self, date: str) -> str:
        """
        Transform date according to specified rules.
        Detect date format, identify month component, convert only that to spoken form.
        
        Args:
            date: Date string to transform (e.g., "11/11/2025" or "2025/11/11")
            
        Returns:
            Transformed date string with only month converted
        """
        # Parse date components
        parts = date.split('/')
        if len(parts) != 3:
            return date
            
        try:
            month_names = self.months.get(self.lang, self.months['it'])  # Default to Italian
            
            part1, part2, part3 = parts
            
            # Detect date format and identify month component
            if len(part3) == 4 and int(part3) > 31:
                # DD/MM/YYYY format - part2 is month
                day, month, year = part1, part2, part3
                month_index = 1  # Month is at index 1 (0-based)
            elif len(part1) == 4 and int(part1) > 31:
                # YYYY/MM/DD format - part2 is month
                year, month, day = part1, part2, part3
                month_index = 1  # Month is at index 1 (0-based)
            else:
                # Default to DD/MM/YYYY format - part2 is month
                day, month, year = part1, part2, part3
                month_index = 1  # Month is at index 1 (0-based)
            
            # Convert only the month component
            transformed_parts = [part1, part2, part3]  # Start with original parts
            if month.isdigit() and 1 <= int(month) <= 12:
                month_name = month_names.get(int(month), month)
                transformed_parts[month_index] = month_name
            
            # Join with spaces (no slashes)
            transformed = " ".join(transformed_parts)
            
            return transformed
            
        except (ValueError, IndexError):
            return date  # Fallback if parsing fails

    def _remove_intermediate_punctuation(self, text: str) -> str:
        """
        Remove intermediate punctuation dots that are not at the end of text and not followed by whitespace.
 
        Args:
            text: Input text to process
 
        Returns:
            Text with intermediate dots removed according to rules
        """
        # Pattern to match dots that are intermediate and not followed by whitespace
        pattern = r'\.(?!\s|$)'
        return re.sub(pattern, '', text)

    def _remove_white_space_before_dot(self, text: str) -> str:
        """
        Remove whitespace before dots in the text.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with whitespace before dots removed
        """
        # Iterate through each dot in the text
        result = []
        i = 0
        while i < len(text):
            if text[i].isspace() and i + 1 < len(text) and text[i + 1] == '.':
                # Skip to append the whitespace
                i += 1
            else:
                result.append(text[i])
                i += 1
        
        return ''.join(result)
    
    def _transform_percent(self, percent: str) -> str:
        """
        Transform percent symbol according to specified rules.
        Only replaces the % symbol with its spoken form, leaving number transformation for later.

        Args:
            percent: Percent string to transform (e.g., "50%" or "25 percent")

        Returns:
            Transformed percent string with % replaced by spoken form
        """
        # Get language-specific mapping
        percent_spoken = self.percent_spoken.get(self.lang, 'percento')

        # Simply replace % with spoken form and space padding
        transformed = re.sub(r'%', f' {percent_spoken} ', percent)

        # Clean up any double spaces that may have been introduced
        transformed = self._remove_double_spaces(transformed)

        return transformed

    def _normalize_unit_symbols(self, text: str) -> str:
        """
        Normalize unit symbols in the text by replacing them with their language-specific spoken forms.

        Args:
            text: Input text to process

        Returns:
            Text with unit symbols replaced by their spoken forms
        """
        transformed = text
        lang = self.lang

        # Get mappings for the current language
        base_units = self.base_units.get(lang, {})
        prefixes = self.unit_prefixes.get(lang, {})
        compound_units = self.compound_units.get(lang, {})

        # First, handle compound units (e.g., 'm/s', 'km/h')
        for compound, spoken in compound_units.items():
            transformed = re.sub(re.escape(compound), f' {spoken} ', transformed)

        # Then, handle prefixed units (e.g., 'km', 'mm', 'MHz')
        # Create a regex pattern for optional prefix + unit
        prefix_pattern = '|'.join(re.escape(p) for p in prefixes.keys()) if prefixes else ''
        unit_pattern = '|'.join(re.escape(u) for u in base_units.keys()) if base_units else ''

        if prefix_pattern and unit_pattern:
            # Use negative lookahead to avoid matching when followed by apostrophe
            # (e.g., "l'" in Italian is a contracted article, not "litri")
            # Require a number before the unit to avoid detecting acronyms like "LT" as units
            pattern = r'(\d+)\s*(' + prefix_pattern + r')?(' + unit_pattern + r')(?!\')(?=\s|$|[^\w])'

            def replace_unit(match):
                number = match.group(1)
                prefix = match.group(2)
                unit = match.group(3)
                spoken = number + ' '
                if prefix:
                    spoken += prefixes.get(prefix.lower(), prefix) + ' '
                # Use lowercase for unit lookup since dictionary keys are lowercase
                spoken += base_units.get(unit.lower(), unit)
                return spoken

            transformed = re.sub(pattern, replace_unit, transformed, flags=re.IGNORECASE)

        # Clean up any double spaces that may have been introduced
        transformed = self._remove_double_spaces(transformed)

        return transformed

    def _analyze_context_for_time(self, text: str, match_start: int, match_end: int) -> str:
        """
        Analyze the context around a matched pattern to determine if it's likely a time or decimal.
        
        Args:
            text: Full text string
            match_start: Start position of the matched pattern
            match_end: End position of the matched pattern
            
        Returns:
            'time' if context suggests it's a time, 'decimal' if suggests it's a decimal, None if unclear
        """
        # Get language-specific keywords
        time_keywords = self.time_keywords.get(self.lang, self.time_keywords['it'])
        time_indicators = time_keywords['time_indicators']
        decimal_indicators = time_keywords['decimal_indicators']
        
        # Extract context window (±3 words around the match)
        words = text.lower().split()
        context_window = 3  # Number of words to check before and after
        
        # Find the word that contains the match
        current_word_index = None
        char_count = 0
        for i, word in enumerate(words):
            word_start = char_count
            word_end = char_count + len(word)
            if word_start <= match_start < word_end:
                current_word_index = i
                break
            char_count = word_end + 1  # +1 for space
        
        if current_word_index is None:
            return None
        
        # Check surrounding words for time/decimal indicators
        start_idx = max(0, current_word_index - context_window)
        end_idx = min(len(words), current_word_index + context_window + 1)
        
        context_words = words[start_idx:end_idx]
        
        # Check for time indicators (words before the pattern)
        for i, word in enumerate(context_words):
            # Clean word of punctuation
            clean_word = re.sub(r'[^\w\'-]', '', word)
            
            # Check for time indicators (especially before the number)
            if clean_word in time_indicators:
                # Additional check: if the word is immediately before or part of time expressions
                if i < len(context_words) // 2:  # Word is before the number
                    return 'time'
            
            # Check for decimal indicators (especially before the number)
            if clean_word in decimal_indicators:
                return 'decimal'
        
        # Check for time range patterns (e.g., "dalle X alle Y")
        text_before = text[:match_start].lower()
        text_after = text[match_end:match_end+50].lower()  # Check 50 chars after
        
        # Look for time prepositions in the text
        time_preps = ['dalle', 'alle', 'da', 'a', 'from', 'to', 'de', 'à', 'von', 'bis']
        for prep in time_preps:
            if prep in text_before:
                return 'time'
        
        # Check for common time patterns
        # If followed by time units like "ore", "hours", etc.
        time_units = ['ore', 'hours', 'heures', 'stunden', 'horas', 'horas', 'uren', 'timmar']
        for unit in time_units:
            if unit in text_after:
                return 'time'
        
        # Default: if no clear indicators, return None (unclear)
        return None

    def _transform_number_with_digits_with_context(self, number: str, match_start: int, match_end: int, full_text: str) -> str:
        """
        Transform number with context analysis to determine if it's a time or decimal.
        
        Args:
            number: Number string to transform
            match_start: Start position of the matched pattern in full text
            match_end: End position of the matched pattern in full text
            full_text: The full text string for context analysis
            
        Returns:
            Transformed number string
        """
        
        # Check if this number is part of a percentage by looking at surrounding context
        # Look for % symbol within a reasonable distance
        context_window = 10  # characters to check before and after
        text_before = full_text[max(0, match_start - context_window):match_start]
        text_after = full_text[match_end:match_end + context_window]
        
        # If there's a % nearby, this is likely a percentage, use decimal context
        if '%' in text_before + text_after:
            context = 'decimal'
        else:
            # Analyze context to determine if this is likely a time or decimal
            context = self._analyze_context_for_time(full_text, match_start, match_end)
            
            # If no clear context is determined, default to 'decimal' for numbers with decimal separators
            if context is None and (',' in number or '.' in number):
                context = 'decimal'
        
        # Transform number with context information
        return self._transform_number_with_digits_with_comma_marker(number, context)

    def _transform_number_with_digits_with_comma_marker(self, number: str, context: str = None) -> str:
        """
        Transform number with digits, handling special comma decimal marker for Italian.
        This is used when a number was originally formatted with comma as decimal separator.
        
        Args:
            number: Number string to transform (can include [COMMA_DECIMAL] marker)
            context: Context hint ('time', 'decimal', or None)
            
        Returns:
            Transformed number string with proper Italian comma handling
        """
        # Check if this number has a comma decimal marker
        has_comma_marker = '[DECIMAL_SEPARATOR]' in number
        if has_comma_marker:
            # Remove the marker for processing
            clean_number = number.replace('[DECIMAL_SEPARATOR]', '')
            # retrieve the correct language decimal separator decimal separator left
            decimal_sep = self.number_formats[self.lang]['decimal_separator']
            spoken_decimal_separator = self.number_formats[self.lang]['decimal_spoken']
            #split on decimal separator
            parts = clean_number.split(decimal_sep)
            if len(parts) == 2:
                integer_part, decimal_part = parts
                result = f"{integer_part} {spoken_decimal_separator} {decimal_part}"
            else:
                # No decimal part, just return the number
                result = clean_number
                # Always remove the marker from the final result
            return result

        else:
            # No comma marker, process normally
            return self._transform_number_with_digits(number, context)

    def _unformat_formatted_numbers(self, text: str) -> str:
        """
        Detect and unformat numbers with decimal separators, removing thousands separators
        to make them suitable for the existing digit processing pipeline.
        Preserves information about decimal separator type for proper spoken conversion.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with formatted numbers converted to clean format with decimal separator info preserved
        """
        # Pattern to match formatted numbers with decimal separators
        # This matches: 50.500,82, 1.234,56, 123.45, etc.
        formatted_number_pattern = r'\b\d+(?:[.,]\d+)+\b'
        
        def unformat_number(match):
            number_str = match.group()

            try:
                # Remove thousands separators
                clean_number = number_str
                for sep in self.number_formats[self.lang]['thousands_separators']:
                    clean_number = clean_number.replace(sep, '')
                # Check if there's a decimal separator left
                decimal_sep = self.number_formats[self.lang]['decimal_separator']
                if decimal_sep in clean_number:
                    # There is a decimal separator
                    return f"{clean_number}[DECIMAL_SEPARATOR]"
                else:
                    # No decimal separator, just return the clean number
                    return clean_number
            except (ValueError, OverflowError):
                # Fallback: if conversion fails, return the original number
                return number_str
        
        # Replace all formatted numbers with unformatted versions
        transformed_text = text
        matches = list(re.finditer(formatted_number_pattern, transformed_text))
        for match in reversed(matches):
            try:
                replacement = unformat_number(match)
            except (ValueError, OverflowError):
                # Fallback: if conversion fails, return the original number
                replacement = match.group()

            transformed_text = transformed_text[:match.start()] + replacement + transformed_text[match.end():]
        
        # Clean up any double spaces that may have been introduced
        transformed_text = self._remove_double_spaces(transformed_text)

        return transformed_text

    def _numbers_to_spoken(self, text: str) -> str:
        """
        Extract digit numbers from text and replace with their language-specific spoken forms.
        Handles standalone numbers (no formatting).
        
        Args:
            text: Input text to process
            
        Returns:
            Text with digit numbers replaced by spoken forms
        """
        # Pattern to match standalone numbers only (no decimal separators)
        # This matches: 123, 4567, but not 50.500,82 or 1.234,56
        number_pattern = r'\b\d+\b'
        
        def replace_number_with_spoken(match):
            number_str = match.group()
            
            try:
                # Handle standalone numbers
                number = int(number_str)
                spoken_number = self._number_to_words(number)
                return spoken_number
            except (ValueError, OverflowError):
                # Fallback: if conversion fails, return the original number
                return number_str
        
        # Replace all numbers in the text with their spoken forms
        transformed_text = text
        matches = list(re.finditer(number_pattern, transformed_text))
        for match in reversed(matches):
            number_str = match.group()

            try:
                # Handle standalone numbers
                number = int(number_str)
                spoken_number = self._number_to_words(number)
                replacement = spoken_number
            except (ValueError, OverflowError):
                # Fallback: if conversion fails, return the original number
                replacement = number_str

            transformed_text = transformed_text[:match.start()] + replacement + transformed_text[match.end():]
        
        # Clean up any double spaces that may have been introduced
        transformed_text = self._remove_double_spaces(transformed_text)
        
        return transformed_text

    def convert_fused_numbers(self, text: str) -> str:
        """
        Convert numbers that are fused with text to their spoken forms.
        Handles numbers embedded within text like 'dell'mh17', 'test123', 'abc123def', etc.
        Adds proper spacing around converted numbers.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with fused numbers replaced by spoken forms with proper spacing
        """
        # Pattern to match numbers fused with text
        # This matches: dell'mh17, test123, abc123def, etc.
        # Use positive lookbehind to ensure we're preceded by start of string or non-digit/non-dot
        # Use positive lookahead to ensure we're followed by end of string or non-digit
        number_pattern = r'(?<!\d)\d+(?!\.\d)'
        
        def replace_fused_number_with_spoken(match):
            number_str = match.group()
            
            try:
                # Handle fused numbers
                number = int(number_str)
                spoken_number = self._number_to_words(number)
                return f" {spoken_number} "  # Add spaces around converted number
            except (ValueError, OverflowError):
                # Fallback: if conversion fails, return the original number
                return f" {number_str} "
        
        # Replace all fused numbers in the text with their spoken forms
        transformed_text = text
        matches = list(re.finditer(number_pattern, transformed_text))
        for match in reversed(matches):
            number_str = match.group()

            try:
                # Handle fused numbers
                number = int(number_str)
                spoken_number = self._number_to_words(number)
                replacement = f" {spoken_number} "  # Add spaces around converted number
            except (ValueError, OverflowError):
                # Fallback: if conversion fails, return the original number
                replacement = f" {number_str} "

            transformed_text = transformed_text[:match.start()] + replacement + transformed_text[match.end():]
        
        # Clean up any double spaces that may have been introduced
        transformed_text = self._remove_double_spaces(transformed_text)
        
        return transformed_text

    def _normalize_unknown_chars(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Normalize characters that the ItalianPhonemesTokenizer doesn't accept.
        Only keep characters that are in the allowed set (matching the tokenizer).
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple of (normalized_text, list of (input_char, output_char) tuples for changed chars)
        """
        import unicodedata
        
        # Punctuation list matching ItalianPhonemesTokenizer
        punct_list = (
            ',', '.', '!', '?', '-',
            ':', ';', '/', '"', '(',
            ')', '[', ']', '{', '}',
            '„', '"', '"', '"', '"', '‒', '—', '«', '»', '‹', '›', '_', ' '
        )
        
        # Allowed IPA characters from ItalianPhonemesTokenizer (must match exactly)
        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyzàèéìòùóæɐɑɔəɚɜɬɹʌʔᵻðŋɛɡɣɪɲɾʃʊʎʒʝβθd͡'t͡'øɒɕɓçɖɘɝɞɟʄɡɠɢʛɦɧħɥʜɨɬɫɮʟɱɯɰɳɵɸœɶʘɺ"
            "ɻʀʁɽʂʈʧʉʋⱱɤʍχʏʑʐʔʡʕʢǀǁǂᵻʃ'ː"
        )
        allowed_chars.update(punct_list)
        
        result = []
        changes = []  # Track (input_char, output_char) for changed chars
        for char in text:
            if char in allowed_chars:
                # Character is valid, keep it as-is
                result.append(char)
            else:
                # Try to normalize unknown characters using NFC
                normalized_char = unicodedata.normalize('NFC', char)
                if normalized_char in allowed_chars:
                    # Normalized form is valid
                    result.append(normalized_char)
                    changes.append((char, normalized_char))
                else:
                    # Try NFD decomposition and check base character
                    decomposed = unicodedata.normalize('NFD', char)
                    if decomposed and decomposed[0] in allowed_chars:
                        # Base character is valid, use it
                        result.append(decomposed[0])
                        changes.append((char, decomposed[0]))
        return ''.join(result), changes

    def _postprocess_text(self, text: str) -> str:
        """
        Postprocess text by replacing % with spoken form, removing hyphens, and cleaning spaces.
        
        Args:
            text: Input text to postprocess
            
        Returns:
            Postprocessed text with % replaced by spoken form, hyphens removed, and spaces cleaned
        """
        # Get language-specific mapping for %
        percent_spoken = self.percent_spoken.get(self.lang, 'percento')
        
        # Step 1: Replace % with its spoken form padded with blank space
        text = re.sub(r'%', f' {percent_spoken} ', text)
        
        # Step 2: Remove all hyphens (-)
        text = text.replace('-', ' ')

        # Step 4: RConvert fused numbers
        text = self.convert_fused_numbers(text)

        text = self._remove_white_space_before_dot(text)
        
        # Step 5: Remove double spaces
        text = self._remove_double_spaces(text)

        return text

    def _preprocess_text(self, text: str, to_lower: bool = True) -> str:
        """
        Preprocess text by extracting and transforming specific patterns.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Step 1: Replace foreign characters with Italian equivalents (for Italian language only)
        text = self._replace_foreign_chars(text)
        
        # Clean up any double spaces that may have been introduced
        text = self._remove_double_spaces(text)
        
        # Extract and transform dates first (before numbers to avoid conflicts)
        # More comprehensive pattern that handles various date formats
        date_pattern = r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b|\b(\d{4})/(\d{1,2})/(\d{1,2})\b'
        matches = list(re.finditer(date_pattern, text))
        for match in reversed(matches):
            transformed = self._transform_date(match.group())
            text = text[:match.start()] + transformed + text[match.end():]
        
        # Extract and transform emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = list(re.finditer(email_pattern, text))
        for match in reversed(matches):
            transformed = self._transform_email(match.group())
            text = text[:match.start()] + transformed + text[match.end():]
        
        # Extract and transform percent symbols FIRST (before numbers to avoid conflicts)
        # Simple pattern to match numbers followed by % symbol
        percent_pattern = r'\d+(?:[.,]\d+)?%'
        matches = list(re.finditer(percent_pattern, text))
        for match in reversed(matches):
            transformed = self._transform_percent(match.group())
            text = text[:match.start()] + transformed + text[match.end():]

        # Transform currency symbols to their spoken forms
        text = self._transform_currency_symbols(text)

        # Normalize unit symbols to their spoken forms
        text = self._normalize_unit_symbols(text)

        # Normalize leading zeros in decimal parts (after percentage processing)
        text = self._normalize_leading_zeros_in_decimals(text)

        # Step 1: Unformat formatted numbers with decimal separators
        # This converts numbers like "50.500,82" to "50500.82" so they can be processed normally
        # This converts the thousand formatted numbert removing the formatting like 28.000 -> 28000
        # Like 28.000.000 -> 28000000
        text = self._unformat_formatted_numbers(text)
        
        # Extract and transform numbers with digits (including negative numbers)
        # More flexible pattern that allows for non-word characters after numbers
        # Exclude numbers that already have the [DECIMAL_SEPARATOR] marker from unformat
        number_pattern = r'-?\d+(?:[.,]\d+)*(?:\[DECIMAL_SEPARATOR\])?(?=\s|$|[^\w])'
        matches = list(re.finditer(number_pattern, text))
        for match in reversed(matches):
            transformed = self._transform_number_with_digits_with_context(match.group(), match.start(), match.end(), text)
            text = text[:match.start()] + transformed + text[match.end():]
        
        # Extract and transform domains
        domain_pattern = r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        matches = list(re.finditer(domain_pattern, text))
        for match in reversed(matches):
            transformed = self._transform_domain(match.group())
            text = text[:match.start()] + transformed + text[match.end():]
        
        # Extract and transform hours (colon format)
        hours_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?\b'
        matches = list(re.finditer(hours_pattern, text))
        for match in reversed(matches):
            transformed = self._transform_hours(match.group())
            text = text[:match.start()] + transformed + text[match.end():]

        # Replace '&' with its spoken form
        and_spoken = self.and_mapping.get(self.lang, 'e')
        text = re.sub(r'&', f' {and_spoken} ', text)

        # Replace special characters with spaces
        text = self._replace_special_chars_with_spaces(text)
        
        # Final cleanup: remove any double spaces that might have been introduced
        text = self._remove_double_spaces(text)
        
        if to_lower:
            return text.lower()
        else:
            return text
    
    def _normalize_ipa_length(self, text: str) -> str:
        out = []
        i = 0
        while i < len(text):
            if i + 1 < len(text) and text[i + 1] == "ː":
                out.append(text[i])
                out.append(text[i])  # gemination
                i += 2
            else:
                if text[i] != "ː":
                    out.append(text[i])
                i += 1
        return "".join(out)
    
    def __call__(self, text: str) -> str:
        """
        Make the Normalizer callable to be compatible with NeMo's text_normalizer_call interface.

        Args:
            text: Input text to normalize
            **kwargs: Additional keyword arguments (passed through to normalize)

        Returns:
            Normalized text string
        """
        return self.normalize(text)
 
    
    def normalize(self, text: str) -> str:
        """
        Normalize text with the required workflow:
        1. Preprocess the text
        2. Normalize the preprocessed text with self.nemo_normalizer.normalize method
        3. Optionally remove final punctuation
        4. Return the result

        Args:
            text: Input text to normalize
            remove_final_punctuation: Whether to remove final punctuation marks (default: True)

        Returns:
            Normalized text string
        """
        # Step 1: Preprocess the text
        preprocessed_text = self._preprocess_text(text, self.to_lower)

        normalized_text = self._numbers_to_spoken(preprocessed_text)

        normalized_text = self._postprocess_text(normalized_text)

        if self.phonemize:
            normalized_text = self.phonemizer.phonemize(text=[normalized_text], strip=True)[0]
            # Normalize any unknown characters from phonemization
            normalized_text, _ = self._normalize_unknown_chars(normalized_text)
            #normalized_text = self._normalize_ipa_length(normalized_text)

        # Step 3: Remove final punctuation if requested
        if not self.tts_mode:
            # Remove common final punctuation marks from the end
            normalized_text = re.sub(r'[.!?;:]+\s*$', '', normalized_text).strip()

        # Step 4: Return the result

        return normalized_text