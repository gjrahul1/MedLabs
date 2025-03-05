// register.js

// Use globally initialized Firebase services (set in firebaseConfig.js)
const auth = window.auth;
const db = window.db;

// Ensure Firebase services are available
if (!auth || !db) {
  console.error("Firebase authentication or Firestore services are not available. Check firebaseConfig.js.");
  alert("Firebase services failed to load. Please refresh the page or contact support.");
  throw new Error("Firebase initialization failed");
}

document.addEventListener('DOMContentLoaded', () => {
  const registerForm = document.getElementById('registration-form');
  if (!registerForm) {
    console.error("Register form not found");
    return;
  }

  // Toggle functionality
  const toggleOptions = document.querySelectorAll('.toggle-option');
  const roleFields = {
    'patient': document.getElementById('patient-fields'),
    'doctor': document.getElementById('doctor-fields'),
    'assistant': document.getElementById('assistant-fields')
  };

  // Validate that all role fields exist
  Object.values(roleFields).forEach((field, index) => {
    if (!field) {
      console.error(`Role field for index ${index} not found in DOM`);
      return;
    }
  });

  function showFields(role) {
    Object.keys(roleFields).forEach(r => {
      const field = roleFields[r];
      field.style.display = r === role ? 'block' : 'none';
      // Remove required attribute from hidden fields to avoid validation errors
      field.querySelectorAll('input[required], select[required]').forEach(input => {
        input.required = (r === role);
      });
    });
    document.getElementById('role').value = role; // Update hidden role field
    toggleOptions.forEach(option => option.classList.toggle('active', option.getAttribute('data-role') === role));
  }

  // Initialize with patient fields
  showFields('patient');

  // Add toggle event listeners
  toggleOptions.forEach(option => {
    option.addEventListener('click', () => {
      const role = option.getAttribute('data-role');
      showFields(role);
    });
  });

  registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const role = document.getElementById('role').value;

    // Dynamically collect data only from the visible role fields
    let userData = {
      role: role
    };

    if (role === 'patient') {
      userData.full_name = document.getElementById('full_name_patient').value.trim();
      userData.email = document.getElementById('email_patient').value.trim();
      userData.phone = document.getElementById('phone_patient').value.trim();
      userData.dob = document.getElementById('dob_patient').value.trim();
      userData.location = document.getElementById('location_patient').value.trim();
      const consultant_id = document.getElementById('consultant_id_patient').value.trim();
      if (consultant_id && consultant_id.match(/^DR00\d+$/)) userData.consultant_id = consultant_id;
      else console.warn(`Invalid consultant_id ${consultant_id} for patient. Skipping.`);
    } else if (role === 'doctor') {
      userData.full_name = document.getElementById('full_name_doctor').value.trim();
      userData.email = document.getElementById('email_doctor').value.trim();
      userData.phone = document.getElementById('phone_doctor').value.trim();
      userData.specialty = document.getElementById('specialty_doctor').value.trim();
      // Generate consultant_id as DR00n
      const consultantId = await generateConsultantId();
      userData.consultant_id = consultantId; // Map doctor_id to consultant_id
      userData.doctor_id = consultantId; // Ensure doctor_id matches consultant_id
    } else if (role === 'assistant') {
      userData.full_name = document.getElementById('full_name_assistant').value.trim();
      userData.email = document.getElementById('email_assistant').value.trim();
      userData.phone = document.getElementById('phone_assistant').value.trim();
      userData.department = document.getElementById('department_assistant').value.trim();
      const assigned_doctors = document.getElementById('assigned_doctors_assistant').value.split(',').map(d => d.trim()).filter(d => d.match(/^DR00\d+$/));
      if (assigned_doctors.length) userData.assigned_doctors = assigned_doctors;
    }

    // Validate required fields for the selected role
    const requiredFields = {
      'patient': ['full_name', 'email', 'phone', 'dob', 'location'],
      'doctor': ['full_name', 'email', 'phone', 'specialty'],
      'assistant': ['full_name', 'email', 'phone', 'department', 'assigned_doctors']
    };

    const missingFields = requiredFields[role].filter(field => !userData[field]);
    if (missingFields.length > 0) {
      alert(`Please fill in all required fields: ${missingFields.join(', ')}`);
      return;
    }

    // Additional validation
    if (!userData.email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
      alert("Please enter a valid email address.");
      return;
    }
    if (!/^\d{10}$/.test(userData.phone)) {
      alert("Please enter a valid 10-digit phone number.");
      return;
    }

    try {
      // Create user in Firebase Auth using email and a temporary password
      const password = "tempPassword123"; // Use a strong default password
      console.log(`Attempting to create user with email: ${userData.email}`);
      const userCredential = await auth.createUserWithEmailAndPassword(userData.email, password);
      const uid = userCredential.user.uid;
      console.log(`User created with UID: ${uid}`);

      // Sign out after creating the user to allow login with chosen password later
      await auth.signOut();
      console.log(`User signed out after creation`);

      // Use the correct collection names
      const collectionMap = {
        'patient': 'patient_registrations',
        'doctor': 'consultant_registrations',
        'assistant': 'assistant_registrations'
      };
      const collectionName = collectionMap[role];

      // Set document ID as consultant_id for doctors, UID for others
      let documentId = uid;
      if (role === 'doctor') {
        documentId = userData.consultant_id; // Use DR00n as document ID
      }

      // Check if user already exists
      const userRef = db.collection(collectionName).doc(documentId);
      const userSnap = await userRef.get();
      if (userSnap.exists) {
        alert(`User with ID ${documentId} already exists!`);
        return;
      }

      // Add UID to userData for reference
      userData.uid = uid;

      // Register user with the generated document ID
      await userRef.set(userData);
      console.log(`User data written to ${collectionName}/${documentId}:`, userData);

      // Verify data in Firestore
      const verifySnap = await userRef.get();
      if (verifySnap.exists) {
        const storedData = verifySnap.data();
        console.log('Verified data in Firestore:', storedData);
        alert(`${role} registered successfully! Please log in with your email and the temporary password: ${password}`);
        window.location.href = '/';
      } else {
        throw new Error('Failed to verify data in Firestore');
      }
    } catch (error) {
      console.error("Registration error:", error);
      alert("Registration failed: " + (error.message || error));
    }
  });

  // Function to generate unique consultant_id (DR00n)
  async function generateConsultantId() {
    const consultantCollection = db.collection('consultant_registrations');
    const snapshot = await consultantCollection.get();
    let maxNumber = 0;

    snapshot.forEach(doc => {
      const id = doc.id;
      const match = id.match(/^DR00(\d+)$/);
      if (match) {
        const num = parseInt(match[1], 10);
        if (!isNaN(num) && num > maxNumber) {
          maxNumber = num;
        }
      } else {
        console.warn(`Document ID ${id} does not follow DR00n format. Skipping.`);
      }
    });

    const newNumber = maxNumber + 1;
    return `DR00${newNumber.toString().padStart(2, '0')}`;
  }
});