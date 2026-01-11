import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bank Stock Price Predictor",
  description: "AI-powered stock price prediction for Indian bank stocks",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
