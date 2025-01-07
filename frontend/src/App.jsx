import { useState } from 'react';
import UploadData from './components/UploadData';
import AnalyzeData from './components/AnalyzeData';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload');

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 p-8">
      <div className="max-w-full mx-auto">
        <h1 className="text-4xl font-extrabold text-center text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 mb-8">
          Data Analysis Dashboard
        </h1>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-gray-800/50 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('upload')}
              className={`px-8 py-2 rounded-md transition-all duration-200 ${
                activeTab === 'upload'
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Upload Data
            </button>
            <button
              onClick={() => setActiveTab('analyze')}
              className={`px-8 py-2 rounded-md transition-all duration-200 ${
                activeTab === 'analyze'
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Analyze Data
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="w-full">
          {activeTab === 'upload' ? <UploadData /> : <AnalyzeData />}
        </div>
      </div>
    </div>
  );
}