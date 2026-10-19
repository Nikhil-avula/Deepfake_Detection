# Using the video checker

If Setup required appears, fix the missing executable or model file first. This is not
a result about the video. Check INSTALLATION.md for Windows PATH and model setup.

1. Open Check a video and choose a short video with one clearly visible person.
2. Press Check this video and wait for the actual check to finish.
3. Read the result and any guidance. Likely real/fake is an estimate, not proof.
4. If the result is Not sure, look for the original source or ask for independent review.
5. If it Cannot analyze, try a clearer front-facing clip or a supported video format.
6. Inspect the face samples and download a report if useful.
7. Use Clear video and result to remove the retained preview and evidence from the session.

Open How reliable is it? for the current measured accuracy, original-video test count,
mistakes and coverage. These measurements do not establish accuracy on arbitrary
internet videos. A strong score can still be wrong. Exact training, validation and
calibration inputs are flagged so they are not mistaken for independent tests.

## When a video cannot be checked

Read the specific warning below the result. The app shows how many face samples
were found and offers preview frames from across the clip. The additional face search needs
at least four sampled face crops and 15% facial coverage. The standard search is
retained and is retried only if its evidence is insufficient. A main foreground person
can have small background faces; several similarly sized people remain ambiguous.
Try a section with a visible face, crop closer to the intended person, or use a
clearer original. Videos with no face are outside this facial detector’s scope.
Setup required means a missing dependency or mismatched model/configuration; follow
INSTALLATION.md rather than changing the video. Replace the entire project when
updating so the trained model, face assets and configuration match.
