from lecflow.database import Lecture
import lecflow.database as database

def test_save_lecture(test_database, tmp_path):
    name = "Test Lecture"
    notes_path = tmp_path / "test_notes.txt"
    transcript_path = tmp_path / "test_transcript.txt"
    lecture_id = database.save_lecture(name, str(notes_path), str(transcript_path))

    session = test_database()
    length = session.query(Lecture).count()
    session.close()

    assert isinstance(lecture_id, int)
    assert length == 1

    

def test_get_all_lectures(test_database, tmp_path):
    name = "Test Lecture"
    notes_path = tmp_path / "test_notes.txt"
    transcript_path = tmp_path / "test_transcript.txt"
    lecture_id = database.save_lecture(name, str(notes_path), str(transcript_path))

    lectures = database.get_all_lectures()

    assert len(lectures) == 1
    assert isinstance(lectures[0], Lecture)
    assert lectures[0].name == name


def test_get_lecture(test_database, tmp_path):
    name = "Test Lecture"
    notes_path = tmp_path / "test_notes.txt"
    transcript_path = tmp_path / "test_transcript.txt"
    lecture_id = database.save_lecture(name, str(notes_path), str(transcript_path))

    lecture = database.get_lecture(lecture_id)

    assert lecture != None
    assert lecture.name == "Test Lecture"
    assert lecture.id == lecture_id
    assert isinstance(lecture, Lecture)


def test_del_lecture(test_database, tmp_path):
    name = "Test Lecture"
    notes_path = tmp_path / "test_notes.txt"
    transcript_path = tmp_path / "test_transcript.txt"
    #Write files because del_lecture will aslos delete the file
    notes_path.write_text("notes")
    transcript_path.write_text("transcript")
    lecture_id = database.save_lecture(name, str(notes_path), str(transcript_path))

    database.del_lecture(lecture_id)

    lecture = database.get_lecture(lecture_id)

    session = test_database()
    length = session.query(Lecture).count()
    session.close()

    assert length == 0
    assert lecture is None
    assert not notes_path.exists()
    assert not transcript_path.exists()
