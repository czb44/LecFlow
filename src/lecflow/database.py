from pathlib import Path
from datetime import datetime, UTC
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

#Initialize SQLite --> lecflow.db
engine = create_engine('sqlite:///db/lecflow.db')

#Initialize base class for models to inherit from
Base = declarative_base()

class Lecture(Base):
    '''Represents a singular lecture / row in lectures table'''
    __tablename__ = 'lectures'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    notes_path = Column(String, nullable=False)
    transcript_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

Base.metadata.create_all(engine) #build table

SessionLocal = sessionmaker(bind=engine) #structure for sessions

def save_lecture(name: str, notes_path: str, transcript_path: str) -> int:
    '''Saves a new lecture to the database, return its id'''
    session = SessionLocal()
    try:
        lecture = Lecture(name=name, notes_path=notes_path, transcript_path=transcript_path)
        session.add(lecture)
        session.commit()
        lecture_id=lecture.id
        return lecture_id
    except Exception:
        session.rollback() #undo unfinished task
        raise #re-raise original error
    finally:
        session.close()

def get_all_lectures() -> list[Lecture]:
    '''Returns all saved lectures'''
    session = SessionLocal()
    try:
        lectures = session.query(Lecture).order_by(Lecture.created_at.desc()).all()
        return lectures
    finally:
        session.close()


def get_lecture(lecture_id: int) -> Lecture | None:
    '''Returns lecture with inputted lecture id or None of lecture DNE'''
    session = SessionLocal()
    try:
        lecture = session.query(Lecture).filter(Lecture.id == lecture_id).first()
        return lecture
    finally:
        session.close()


def del_lecture(lecture_id: int) -> None:
    '''Deletes a lecture and associated files by id if exists'''
    session = SessionLocal()
    try:
        lecture = session.query(Lecture).filter(Lecture.id == lecture_id).first()
        if not lecture: #skip if lecture DNE
            return
        
        notes_path = Path(lecture.notes_path)
        transcript_path = Path(lecture.transcript_path)
        if notes_path.exists():
            notes_path.unlink()
        if transcript_path.exists():
            transcript_path.unlink()
        
        session.delete(lecture)
        session.commit()
    except Exception:
        session.rollback() #undo unfinished task
        raise #re-raise original error
    finally: #ensure database session closes on failures
        session.close()


if __name__ == '__main__':
    save_lecture('test_lecture', 'outputs/notes/test.md', 'outputs/transcripts/test.txt')
    lectures = get_all_lectures()
    for lec in lectures:
        print(lec.id, lec.name, lec.created_at)
