RAG SMART ATTENDANCE - VS CODE LIVE DEMO

1. Extract the ZIP.
2. Open the folder in VS Code.
3. Install the VS Code extension "Live Server".
4. Right click index.html -> Open with Live Server.
5. Login: admin / admin123
6. Register students first: Student ID, Roll, Name, Class, Section, then Start System Camera and Capture Face.
7. Attendance: Start System Camera -> choose Morning/Afternoon -> Scan & Verify Face. Only a registered face match is accepted.
8. Session rules: Start to Late After = Present; Late After to End = Late; no record by End = Absent in reports.
9. Duplicate attendance for the same student/date/session is rejected.
10. Monthly Report provides Morning/Afternoon and status totals; CSV export is available.
11. RAG Search retrieves relevant stored rules and historical student attendance from browser storage.

IMPORTANT: Face matching uses face-api.js loaded from a CDN and is suitable for an academic demo, not a production biometric system. A real deployment needs secure backend storage, validated liveness/anti-spoofing, consent, encryption, and proper biometric governance. Camera access requires localhost/HTTPS.
