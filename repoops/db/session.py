from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url
import socket
from sqlalchemy.orm import sessionmaker
from repoops.common.config import DATABASE_URL

def _host_resolves(host: str) -> bool:
    try:
        socket.getaddrinfo(host, None)
        return True
    except Exception:
        return False


def _coerce_db_url(url: str) -> str:
    if not url:
        return url
    # If psycopg2 is referenced but not installed, fall back to psycopg3.
    if "postgresql+psycopg2://" in url:
        try:
            import psycopg2  # noqa: F401
            return url
        except Exception:
            return url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)

    # If no driver is specified, prefer psycopg3 when available.
    if url.startswith("postgresql://"):
        try:
            import psycopg  # noqa: F401
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        except Exception:
            return url

    # If running outside docker and host is "db", fall back to localhost.
    try:
        url_obj = make_url(url)
        if url_obj.host == "db" and not _host_resolves("db"):
            url_obj = url_obj.set(host="localhost")
            return str(url_obj)
    except Exception:
        pass

    return url

engine = create_engine(_coerce_db_url(DATABASE_URL), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
