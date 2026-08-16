from datetime import datetime
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
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine) #build table

SessionLocal = sessionmaker(bind=engine) #structure for sessions

def save_lecture(name: str, notes_path: str, transcript_path: str) -> int:
    '''Saves a new lecture to the database, return its id'''
    session = SessionLocal()
    lecture = Lecture(name=name, notes_path=notes_path, transcript_path=transcript_path)
    session.add(lecture)
    session.commit()
    lecture_id=lecture.id
    session.close()
    return lecture_id

def get_all_lectures() -> list[Lecture]:
    '''Returns all saved lectures'''
    session = SessionLocal()
    lectures = session.query(Lecture).order_by(Lecture.created_at.desc()).all()
    session.close()
    return lectures

def get_lecture(lecture_id: int) -> Lecture | None:
    '''Returns lecture with inputted lecture id or None of lecture DNE'''
    session = SessionLocal()
    lecture = session.query(Lecture).filter(Lecture.id == lecture_id).first()
    session.close()
    return lecture

def del_lecture(lecture_id: int) -> None:
    '''Deletes a lecture by id if lecture exists'''
    session = SessionLocal()
    lecture = session.query(Lecture).filter(Lecture.id == lecture_id).first()
    if lecture: #ensure exist
        session.delete(lecture)
        session.commit()
    session.close()


if __name__ == '__main__':
    save_lecture('test_lecture', 'outputs/notes/test.md', 'outputs/transcripts/test.txt')
    lectures = get_all_lectures()
    for lec in lectures:
        print(lec.id, lec.name, lec.created_at)
