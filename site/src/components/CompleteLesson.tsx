import { useEffect, useState } from "react";
import { getProgressStore } from "../lib/progress";
import "./components.css";

interface Props {
  lessonId: string;
}

export default function CompleteLesson({ lessonId }: Props) {
  const [done, setDone] = useState(false);

  useEffect(() => {
    const store = getProgressStore();
    // A change that arrives while the first load is in flight is newer than that load.
    let fresh = true;
    store.load().then((progress) => fresh && setDone(Boolean(progress.lessons[lessonId])));
    const unsubscribe = store.subscribe((progress) => {
      fresh = false;
      setDone(Boolean(progress.lessons[lessonId]));
    });
    return () => {
      fresh = false;
      unsubscribe();
    };
  }, [lessonId]);

  return (
    <div className="complete" role="status">
      {done ? (
        <span className="complete-done">Completed</span>
      ) : (
        <button type="button" className="btn btn-primary" onClick={() => void getProgressStore().completeLesson(lessonId)}>
          Mark lesson complete
        </button>
      )}
    </div>
  );
}
