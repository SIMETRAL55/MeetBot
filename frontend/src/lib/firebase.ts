/**
 * Firebase app initialization.
 *
 * All imports are lazy so Firebase is only loaded when actually used
 * (e.g., social login). The config is driven entirely by environment
 * variables so no secrets appear in source.
 */

let _app: import("firebase/app").FirebaseApp | null = null;

export async function getFirebaseApp() {
  if (_app) return _app;

  const { initializeApp, getApps } = await import("firebase/app");

  const existingApps = getApps();
  if (existingApps.length > 0) {
    _app = existingApps[0];
    return _app;
  }

  const config = {
    apiKey: process.env.NEXT_PUBLIC_FIREBASE_WEB_API_KEY,
    authDomain: `${process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID}.firebaseapp.com`,
    projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  };

  _app = initializeApp(config);
  return _app;
}

export async function getFirebaseAuth() {
  const { getAuth } = await import("firebase/auth");
  const app = await getFirebaseApp();
  return getAuth(app);
}

export async function signInWithGoogle(): Promise<string | null> {
  try {
    const { GoogleAuthProvider, signInWithPopup } = await import("firebase/auth");
    const auth = await getFirebaseAuth();
    const provider = new GoogleAuthProvider();
    const result = await signInWithPopup(auth, provider);
    return await result.user.getIdToken();
  } catch (err) {
    console.error("Firebase Google sign-in failed:", err);
    return null;
  }
}

export async function sendVerificationEmail(): Promise<void> {
  const { sendEmailVerification } = await import("firebase/auth");
  const auth = await getFirebaseAuth();
  const user = auth.currentUser;
  if (!user) throw new Error("No authenticated Firebase user");
  await sendEmailVerification(user);
}

/**
 * Reload the current Firebase user and return a fresh ID token if their email
 * is now verified, or null if still unverified / no user present.
 * Pass the returned token directly to api.firebaseLogin().
 */
export async function registerWithEmail(
  email: string,
  password: string
): Promise<string> {
  const { createUserWithEmailAndPassword, sendEmailVerification } =
    await import("firebase/auth");
  const auth = await getFirebaseAuth();
  const result = await createUserWithEmailAndPassword(auth, email, password);
  // Send verification email immediately after creation
  await sendEmailVerification(result.user);
  // Return the Firebase ID token so backend can sync the user
  return result.user.getIdToken();
}

export async function sendPasswordReset(email: string): Promise<void> {
  const { sendPasswordResetEmail } = await import("firebase/auth");
  const auth = await getFirebaseAuth();
  // Firebase throws auth/user-not-found — catch and ignore to prevent enumeration
  try {
    await sendPasswordResetEmail(auth, email);
  } catch (err: unknown) {
    if ((err as { code?: string }).code === "auth/user-not-found") return;
    throw err;
  }
}

export async function reloadAndCheckVerified(): Promise<string | null> {
  const auth = await getFirebaseAuth();
  const user = auth.currentUser;
  if (!user) return null;
  await user.reload();
  if (!user.emailVerified) return null;
  return user.getIdToken(true);
}
