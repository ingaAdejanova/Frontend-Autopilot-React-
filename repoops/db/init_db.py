from repoops.db.session import engine
from repoops.db.models import Base

def init_db():
    Base.metadata.create_all(bind=engine)
