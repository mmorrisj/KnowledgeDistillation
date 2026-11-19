import { useEffect, useState } from "react";
import { getProgressStore, percentComplete, type Progress } from "../lib/progress";
import "./components.css";

interface Props {
  lessonIds: string[];
}

export default function ProgressBar({ lessonIds }: Props) {
  const [progress, setProgress] = useState<Progress>({ lessons: {}, quizzes: {} });

  useEffect(() => {
    const store = getProgressStore();
    // A change that arrives while the first load is in flight is newer than that load.
    let fresh = true;
    store.load().then((loaded) => fresh && setProgress(loaded));
    const unsubscribe = store.subscribe((next) => {
      fresh = false;
      setProgress(next);
    });
    return () => {
      fresh = false;
      unsubscribe();
    };
  }, []);

  useEffect(() => {
    // Tick completed lessons in the sidebar, which is plain HTML outside this component.
    document.querySelectorAll<HTMLElement>("[data-lesson-id]").forEach((link) => {
      const complete = Boolean(progress.lessons[link.dataset.lessonId ?? ""]);
      link.dataset.complete = String(complete);
    });
  }, [progress]);

  const percent = percentComplete(progress, lessonIds);
  const done = lessonIds.filter((id) => progress.lessons[id]).length;

  return (
    <div className="progress">
      <div
        className="progress-track"
        role="progressbar"
        aria-label="Course progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
      >
        <div className="progress-fill" style={{ width: `${percent}%` }} />
      </div>
      <p className="progress-label">
        {done} of {lessonIds.length} lessons complete
      </p>
    </div>
  );
}
