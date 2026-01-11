import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const filePath = path.join(process.cwd(), '..', 'future_predictions_next_day.json');
    const fileContents = fs.readFileSync(filePath, 'utf8');
    const predictions = JSON.parse(fileContents);

    // Sort by predicted change percentage (highest gains first)
    predictions.sort((a: any, b: any) => b.Predicted_Change_Pct - a.Predicted_Change_Pct);

    return NextResponse.json(predictions);
  } catch (error) {
    console.error('Error reading predictions:', error);
    return NextResponse.json(
      { error: 'Failed to load predictions' },
      { status: 500 }
    );
  }
}
