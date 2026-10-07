# soap_client.py
# ─────────────────────────────────────────────────────────────────────
# Thin wrapper around the SOAP API that persists matched story pairs.
#
# Reads DATA_SAVE_API_URL from environment / .env file.
# The zeep Client is created ONCE at module load and reused — creating
# a new Client per call is expensive (downloads and parses the WSDL).
# ─────────────────────────────────────────────────────────────────────

import logging
import os

from dotenv import load_dotenv
from zeep import Client
from zeep.exceptions import Fault, TransportError

load_dotenv()

logger = logging.getLogger(__name__)

_WSDL_URL: str = os.getenv("DATA_SAVE_API_URL", "")

if not _WSDL_URL:
    raise EnvironmentError(
        "DATA_SAVE_API_URL is not set. "
        "Add it to your .env file or environment before starting the app."
    )

logger.info("Initialising SOAP client from WSDL: %s", _WSDL_URL)
_soap_client = Client(_WSDL_URL)
logger.info("SOAP client ready. Services: %s", list(_soap_client.wsdl.services.keys()))


class SOAPCallError(RuntimeError):
    """Raised when the SOAP API returns a fault or a transport error."""
    pass


def save_story_pair(story_id_1: str, story_id_2: str) -> None:
    """
    Call the SOAP API to persist a matched story pair.

    Parameters
    ----------
    story_id_1 : str
        The incoming story's ID (caller's own ID, not the Qdrant UUID).
    story_id_2 : str
        The matched story's ID (caller's own ID, not the Qdrant UUID).

    Raises
    ------
    SOAPCallError
        If the SOAP service returns a fault or a network/transport error occurs.
        Any other unexpected exception is also wrapped and re-raised so the
        caller can treat all SOAP failures uniformly.
    """
    try:
        logger.info("SOAP save_story_pair: story1='%s'  story2='%s'", story_id_1, story_id_2)
        _soap_client.service.SaveCustomer(story1=story_id_1, story2=story_id_2)
        logger.info("SOAP call succeeded for pair ('%s', '%s')", story_id_1, story_id_2)

    except Fault as e:
        msg = f"SOAP fault for pair ('{story_id_1}', '{story_id_2}'): {e}"
        logger.error(msg)
        raise SOAPCallError(msg) from e

    except TransportError as e:
        msg = f"SOAP transport error for pair ('{story_id_1}', '{story_id_2}'): {e}"
        logger.error(msg)
        raise SOAPCallError(msg) from e

    except Exception as e:
        msg = f"Unexpected SOAP error for pair ('{story_id_1}', '{story_id_2}'): {e}"
        logger.error(msg)
        raise SOAPCallError(msg) from e
