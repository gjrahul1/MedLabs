// login.js (Login Flow Only)

// Ensure Firebase services are available
if (!window.auth || !window.db) {
  console.error("Firebase not initialized. Check your firebaseConfig.js file.");
  alert("Firebase initialization failed. Please refresh the page or check the console for details.");
} else {
  console.log("✅ Firebase initialized in login.js");
}

const auth = window.auth;
const db = window.db;

function showFields(role) {
  document.getElementById('role').value = role;
  const toggleOptions = document.querySelectorAll('.toggle-option');
  toggleOptions.forEach(option => {
    option.classList.toggle('active', option.getAttribute('data-role') === role);
  });

  const fields = {
    'patient': document.getElementById('patient-fields'),
    'doctor': document.getElementById('doctor-fields'),
    'assistant': document.getElementById('assistant-fields')
  };

  // Hide all fields and remove required attributes
  Object.values(fields).forEach(field => {
    field.style.display = 'none';
    field.querySelectorAll('input[required]').forEach(input => {
      input.removeAttribute('required');
    });
  });

  // Show selected role fields and add required attributes
  const selectedField = fields[role];
  if (selectedField) {
    selectedField.style.display = 'block';
    selectedField.querySelectorAll('input').forEach(input => {
      if (input.placeholder.includes('Email') || input.placeholder.includes('Password')) {
        input.setAttribute('required', '');
      }
    });
  }
}

async function login(event) {
  event.preventDefault();

  const role = document.getElementById('role').value;
  let email, password;

  // Capture email and password based on role
  if (role === 'patient') {
    email = document.getElementById('email').value.trim();
    password = document.getElementById('password').value.trim();
  } else if (role === 'doctor') {
    email = document.getElementById('doctor_id').value.trim();
    password = document.getElementById('doctor_password').value.trim();
  } else if (role === 'assistant') {
    email = document.getElementById('lab_id').value.trim();
    password = document.getElementById('lab_password').value.trim();
  } else {
    alert('Invalid role selected.');
    console.error('Invalid role:', role);
    return;
  }

  // Log inputs for debugging
  console.log('Role:', role);
  console.log('Email:', email);
  console.log('Password:', password ? '[REDACTED]' : 'Missing');

  // Validate inputs
  if (!email || !password) {
    alert('Please provide both email and password.');
    console.error('Missing email or password');
    return;
  }

  try {
    // Authenticate user via Firebase Authentication
    console.log(`Attempting login for ${email} as ${role}...`);
    const userCredential = await auth.signInWithEmailAndPassword(email, password);
    const user = userCredential.user;
    if (!user) {
      throw new Error('Firebase Authentication returned no user.');
    }
    console.log('User authenticated:', user.uid, user.email);

    // Retrieve a fresh ID token
    const idToken = await user.getIdToken(true);
    console.log('Fetched ID Token:', idToken);

    // Verify role in Firestore using email
    const collectionMap = {
      patient: 'patient_registrations',
      doctor: 'consultant_registrations',
      assistant: 'assistant_registrations'
    };
    const collectionName = collectionMap[role];
    const userQuery = await db.collection(collectionName).where('email', '==', email.toLowerCase()).limit(1).get();
    if (userQuery.empty) {
      await auth.signOut();
      throw new Error(`User not registered as a ${role}. Please register first.`);
    }

    const userDoc = userQuery.docs[0];
    const userData = userDoc.data();
    const documentId = userData.consultant_id || user.uid; // Use consultant_id for doctors, fallback to uid
    console.log(`Role verified in Firestore: ${role}, Document ID: ${documentId}`);

    // Send ID token, role, and uid to backend
    const response = await fetch('/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${idToken}`
      },
      body: JSON.stringify({ role, email, uid: user.uid })
    });

    if (response.redirected) {
      window.location.href = response.url;
    } else if (response.ok) {
      const data = await response.json();
      sessionStorage.setItem('idToken', idToken);
      console.log('Stored ID Token in sessionStorage:', idToken);
      window.location.href = '/dashboard';
    } else {
      const errorData = await response.json();
      throw new Error(`Login Failed: ${errorData.error}`);
    }
  } catch (error) {
    console.error('Login Error:', error);
    if (auth.currentUser) {
      await auth.signOut();
    }
    // Specific Firebase error handling
    if (error.code === 'auth/invalid-login-credentials') {
      alert('Invalid email or password. Please check your credentials and try again.');
    } else if (error.code === 'auth/user-not-found') {
      alert('No user found with this email. Please register first.');
    } else if (error.code === 'auth/wrong-password') {
      alert('Incorrect password. Please try again.');
    } else {
      alert(`Authentication Error: ${error.message}`);
    }
  }
}

async function logout() {
  try {
    await auth.signOut();
    sessionStorage.removeItem('idToken');
    window.location.href = '/logout';
  } catch (error) {
    console.error('Logout Error:', error);
    alert('Logout failed. Please try again.');
  }
}

window.onload = () => {
  showFields('patient');
  const loginForm = document.getElementById('login-form');
  if (loginForm) {
    loginForm.addEventListener('submit', login);
  }
};