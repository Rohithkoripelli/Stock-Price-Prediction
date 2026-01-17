'use client';

import { useEffect, useState } from 'react';
import ChatInterface from '@/components/ChatInterface';

interface StockPrediction {
  Stock: string;
  Ticker: string;
  Current_Price: number;
  Predicted_Direction: 'UP' | 'DOWN';
  Direction_Confidence: number;
  Predicted_Change_Pct: number;
  Predicted_Price_Low: number;
  Predicted_Price_Mid: number;
  Predicted_Price_High: number;
  Range_Pct: number;
  Potential_Gain_Loss_Mid: number;
}

export default function Home() {
  const [predictions, setPredictions] = useState<StockPrediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/predictions')
      .then(res => res.json())
      .then(data => {
        setPredictions(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error fetching predictions:', err);
        setLoading(false);
      });
  }, []);

  const getSignalCategory = (prediction: StockPrediction) => {
    if (prediction.Predicted_Direction === 'UP' && prediction.Direction_Confidence > 60) {
      return { label: 'STRONG BUY', color: 'bg-green-500', textColor: 'text-green-500' };
    } else if (prediction.Predicted_Direction === 'UP' && prediction.Direction_Confidence >= 50) {
      return { label: 'MODERATE BUY', color: 'bg-yellow-500', textColor: 'text-yellow-500' };
    } else if (prediction.Predicted_Direction === 'DOWN' && prediction.Direction_Confidence > 60) {
      return { label: 'STRONG SELL', color: 'bg-red-500', textColor: 'text-red-500' };
    } else if (prediction.Predicted_Direction === 'DOWN' && prediction.Direction_Confidence >= 50) {
      return { label: 'MODERATE SELL', color: 'bg-orange-500', textColor: 'text-orange-500' };
    } else {
      return { label: 'HOLD', color: 'bg-gray-500', textColor: 'text-gray-500' };
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-300">Loading predictions...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-800 dark:text-white mb-4">
            Bank Stock Price Predictor
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            AI-Powered Next-Day Price Predictions for Indian Bank Stocks
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
            Using Advanced Transformer Models trained on historical data
          </p>
        </div>

        {/* Stats Summary */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
          {[
            { label: 'Strong Buy', count: predictions.filter(p => p.Predicted_Direction === 'UP' && p.Direction_Confidence > 60).length, color: 'bg-green-500' },
            { label: 'Moderate Buy', count: predictions.filter(p => p.Predicted_Direction === 'UP' && p.Direction_Confidence >= 50 && p.Direction_Confidence <= 60).length, color: 'bg-yellow-500' },
            { label: 'Strong Sell', count: predictions.filter(p => p.Predicted_Direction === 'DOWN' && p.Direction_Confidence > 60).length, color: 'bg-red-500' },
            { label: 'Moderate Sell', count: predictions.filter(p => p.Predicted_Direction === 'DOWN' && p.Direction_Confidence >= 50 && p.Direction_Confidence <= 60).length, color: 'bg-orange-500' },
            { label: 'Hold', count: predictions.filter(p => p.Direction_Confidence < 50).length, color: 'bg-gray-500' },
          ].map((stat, idx) => (
            <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-md">
              <div className={`${stat.color} w-3 h-3 rounded-full mb-2`}></div>
              <p className="text-2xl font-bold text-gray-800 dark:text-white">{stat.count}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">{stat.label}</p>
            </div>
          ))}
        </div>

        {/* Chat Interface */}
        <div className="mb-8">
          <ChatInterface />
        </div>

        {/* Predictions Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {predictions.map((prediction, idx) => {
            const signal = getSignalCategory(prediction);
            return (
              <div key={idx} className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden hover:shadow-2xl transition-shadow duration-300">
                {/* Card Header */}
                <div className={`${signal.color} p-4`}>
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="text-white font-bold text-lg">{prediction.Stock}</h3>
                      <p className="text-white text-sm opacity-90">{prediction.Ticker}</p>
                    </div>
                    <div className="bg-white bg-opacity-20 rounded-lg px-3 py-1">
                      <p className="text-white text-xs font-semibold">{signal.label}</p>
                    </div>
                  </div>
                </div>

                {/* Card Body */}
                <div className="p-6">
                  {/* Current Price */}
                  <div className="mb-4">
                    <p className="text-sm text-gray-500 dark:text-gray-400">Current Price</p>
                    <p className="text-3xl font-bold text-gray-800 dark:text-white">
                      ₹{prediction.Current_Price.toFixed(2)}
                    </p>
                  </div>

                  {/* Prediction */}
                  <div className="mb-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <p className="text-sm font-semibold text-gray-700 dark:text-gray-300">Prediction</p>
                      <span className={`${signal.textColor} font-bold flex items-center`}>
                        {prediction.Predicted_Direction === 'UP' ? '↑' : '↓'} {prediction.Predicted_Direction}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <p className="text-xs text-gray-500 dark:text-gray-400">Confidence</p>
                      <p className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                        {prediction.Direction_Confidence.toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {/* Price Change */}
                  <div className="mb-4">
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Expected Change</p>
                    <p className={`text-xl font-bold ${prediction.Predicted_Change_Pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {prediction.Predicted_Change_Pct >= 0 ? '+' : ''}{prediction.Predicted_Change_Pct.toFixed(2)}%
                    </p>
                    <p className={`text-sm ${prediction.Potential_Gain_Loss_Mid >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      ₹{prediction.Potential_Gain_Loss_Mid >= 0 ? '+' : ''}{prediction.Potential_Gain_Loss_Mid.toFixed(2)}
                    </p>
                  </div>

                  {/* Price Range */}
                  <div className="border-t border-gray-200 dark:border-gray-600 pt-4">
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Predicted Price Range</p>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Low</span>
                        <span className="font-semibold text-gray-800 dark:text-white">₹{prediction.Predicted_Price_Low.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Mid</span>
                        <span className="font-bold text-gray-800 dark:text-white">₹{prediction.Predicted_Price_Mid.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">High</span>
                        <span className="font-semibold text-gray-800 dark:text-white">₹{prediction.Predicted_Price_High.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Trading Recommendation */}
                  <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900 dark:bg-opacity-20 rounded-lg">
                    <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Recommendation</p>
                    {signal.label === 'STRONG BUY' && (
                      <p className="text-xs text-gray-700 dark:text-gray-300">
                        High confidence upward prediction. Consider buying with stop loss at ₹{(prediction.Predicted_Price_Low * 0.98).toFixed(2)}
                      </p>
                    )}
                    {signal.label === 'MODERATE BUY' && (
                      <p className="text-xs text-gray-700 dark:text-gray-300">
                        Moderate confidence. Use smaller position size.
                      </p>
                    )}
                    {signal.label === 'STRONG SELL' && (
                      <p className="text-xs text-gray-700 dark:text-gray-300">
                        High confidence downward prediction. Avoid buying or consider exiting long positions.
                      </p>
                    )}
                    {signal.label === 'MODERATE SELL' && (
                      <p className="text-xs text-gray-700 dark:text-gray-300">
                        Moderate downward signal. Consider reducing exposure.
                      </p>
                    )}
                    {signal.label === 'HOLD' && (
                      <p className="text-xs text-gray-700 dark:text-gray-300">
                        Low confidence. Wait for clearer signal.
                      </p>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Disclaimer */}
        <div className="mt-12 bg-yellow-50 dark:bg-yellow-900 dark:bg-opacity-20 border-l-4 border-yellow-400 p-4 rounded">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            <strong>Disclaimer:</strong> These predictions are generated by AI models trained on historical data.
            Past performance does not guarantee future results. Use these predictions as ONE input in your trading
            decisions. Always do your own research and consult with a financial advisor before making investment decisions.
          </p>
        </div>
      </div>
    </div>
  );
}
