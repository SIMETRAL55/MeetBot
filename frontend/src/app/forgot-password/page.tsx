import { Mic } from "lucide-react";
import ForgotPasswordForm from "@/components/auth/ForgotPasswordForm";

export const metadata = { title: "Forgot Password — MeetBot" };

export default function ForgotPasswordPage() {
  return (
    <div className="flex min-h-[calc(100vh-8rem)] items-center justify-center px-4">
      <div className="w-full max-w-sm space-y-8">
        <div className="flex flex-col items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-amber-500/10 ring-1 ring-amber-500/25">
            <Mic className="h-6 w-6 text-amber-400" />
          </div>
          <span className="font-display text-lg font-semibold text-white">MeetBot</span>
        </div>
        <ForgotPasswordForm />
      </div>
    </div>
  );
}
