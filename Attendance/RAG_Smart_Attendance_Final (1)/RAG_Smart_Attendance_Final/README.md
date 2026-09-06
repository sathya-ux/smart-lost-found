# RAG Smart Attendance Final

## Run in VS Code
1. Extract ZIP.
2. Open folder in VS Code.
3. Install **Live Server** extension.
4. Right-click `index.html` -> **Open with Live Server**.
5. Login:
   - Admin: `admin` / `admin123`
   - Staff: `staff` / `staff123`

## Main flow
Admin registers student -> Student ID/Roll/Name/Department/Class/Section -> System Camera face capture -> face descriptor saved -> Live Attendance scans face -> nearest registered face is matched -> unknown face rejected -> session timing checked -> Present/Late marked -> duplicate same day/session rejected.

Morning and Afternoon are separate. Admin can change Start, Late After and End time.

Staff dashboard is department scoped using the configured staff department.

Monthly report separates Morning and Afternoon and calculates a session percentage. CSV export is included.

RAG Search retrieves relevant student profile and historical attendance from the browser's local data.

## Camera / face recognition
Camera access requires `localhost`/Live Server or HTTPS and browser camera permission. The demo uses `face-api.js` from a CDN. Production use requires secure backend storage, validated liveness/anti-spoofing, consent, encryption, and privacy controls.
