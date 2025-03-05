// storage.js for pdf files

// Ensure Firebase is initialized (import this after firebase-config.js)
const storage = firebase.storage();
const db = window.db; // Reference to Firestore from firebaseConfig.js

// Function to upload files (prescriptions or lab records)
async function uploadFile(file, type) {
    const user = firebase.auth().currentUser;
    if (!user) return alert("User not logged in");

    try {
        // Get the user's document using email to find the correct uid
        const userQuery = await db.collection('patient_registrations').where('email', '==', user.email).limit(1).get();
        if (userQuery.empty) {
            alert("User not found in patient database.");
            return;
        }

        const userDoc = userQuery.docs[0];
        const userData = userDoc.data();
        const patientId = userData.uid; // Use the uid from the document

        // Determine folder based on file type
        const folder = type === "prescription" ? "prescriptions" : "lab_records";
        const fileRef = storage.ref(`patients/${patientId}/${folder}/${file.name}`);

        // Upload file to Firebase Storage
        const snapshot = await fileRef.put(file);
        const downloadURL = await snapshot.ref.getDownloadURL();

        // Update Firestore with file reference
        await db.collection('patient_registrations')
            .doc(patientId)
            .collection(folder)
            .add({
                fileName: file.name,
                fileURL: downloadURL,
                uploadedAt: firebase.firestore.FieldValue.serverTimestamp(),
            });

        alert(`${type} uploaded successfully!`);
    } catch (error) {
        console.error("Upload failed: ", error);
        alert("Error uploading file: " + error.message);
    }
}

// Event listeners for prescription and lab record uploads
document.getElementById('prescription-upload')?.addEventListener('change', (e) => uploadFile(e.target.files[0], "prescription"));
document.getElementById('lab-upload')?.addEventListener('change', (e) => uploadFile(e.target.files[0], "lab"));