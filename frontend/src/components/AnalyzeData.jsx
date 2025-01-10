import { useState } from 'react';
import { AlertCircle, Save, Send, ChevronDown } from 'lucide-react';

const AnalyzeData = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [isQueryExpanded, setIsQueryExpanded] = useState(true);

  const handleAnalysis = async (e) => {
    e.preventDefault();
    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('https://nltosql-hxdwg2arhdhja4dt.canadacentral-01.azurewebsites.net/api/analysis/', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query }),
      });

      const data = await response.json();

      if (response.ok) {
        setResults(data);
      } else {
        throw new Error(data.error || 'Error analyzing data');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveResults = async () => {
    if (!results?.results) return;
  
    try {
      const response = await fetch('https://nltosql-hxdwg2arhdhja4dt.canadacentral-01.azurewebsites.net/api/save-results/', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ results: results.results }),
      });
  
      if (!response.ok) throw new Error('Failed to download results');
  
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      
      const contentDisposition = response.headers.get('Content-Disposition');
      const filename = contentDisposition
        ? contentDisposition.split('filename=')[1].replace(/"/g, '')
        : 'query_results.csv';
      
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="flex w-full h-[calc(100vh-12rem)] bg-gray-800/50 rounded-xl overflow-hidden">
      {/* Left Panel - Query Input */}
      <div className="w-1/2 p-6 border-r border-gray-700">
        <form onSubmit={handleAnalysis} className="space-y-4">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your query in natural language..."
            className="w-full p-4 border border-gray-600 rounded-lg bg-gray-800 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent h-40 resize-none"
          />
          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-blue-500 to-indigo-500 text-white py-3 px-4 rounded-lg font-medium hover:from-blue-600 hover:to-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          >
            {loading ? (
              'Analyzing...'
            ) : (
              <>
                <Send size={20} /> Analyze Data
              </>
            )}
          </button>
        </form>

        {error && (
          <div className="mt-4 flex items-start gap-2 bg-red-500/10 border border-red-500/50 rounded-lg p-4 text-red-400">
            <AlertCircle className="mt-1 flex-shrink-0" size={20} />
            <p>{error}</p>
          </div>
        )}
      </div>

      {/* Right Panel - Results */}
      <div className="w-1/2 p-6 overflow-y-auto">
        {results && (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h3 className="text-2xl font-bold text-white">Analysis Results</h3>
              <button
                onClick={handleSaveResults}
                className="flex items-center gap-2 bg-green-500 text-white py-2 px-4 rounded-lg hover:bg-green-600 transition-all duration-200"
              >
                <Save size={20} />
                Save Results
              </button>
            </div>

            {results.explanation && (
              <div className="bg-gray-800 rounded-lg border border-gray-600 overflow-hidden">
                <div className="px-4 py-3 bg-gray-700 border-b border-gray-600">
                  <h4 className="font-semibold text-white">Explanation</h4>
                </div>
                <div className="p-4">
                  <div dangerouslySetInnerHTML={{ __html: results.explanation }} className="text-gray-200 prose prose-invert" />
                </div>
              </div>
            )}

            {results.query && (
              <div className="bg-gray-800 rounded-lg border border-gray-600 overflow-hidden">
                <button
                  onClick={() => setIsQueryExpanded(!isQueryExpanded)}
                  className="w-full px-4 py-3 bg-gray-700 flex justify-between items-center hover:bg-gray-600 transition-colors duration-200"
                >
                  <h4 className="font-semibold text-white">Generated Query</h4>
                  <ChevronDown
                    size={20}
                    className={`text-white transform transition-transform duration-200 ${
                      isQueryExpanded ? 'rotate-180' : ''
                    }`}
                  />
                </button>
                <div
                  className={`transition-all duration-200 ${
                    isQueryExpanded ? 'max-h-96' : 'max-h-0'
                  } overflow-hidden`}
                >
                  <div className="p-4">
                    <code className="block whitespace-pre-wrap text-blue-400 font-mono text-sm">
                      {results.query}
                    </code>
                  </div>
                </div>
              </div>
            )}

            {results.results && (
              <div className="bg-gray-800 rounded-lg border border-gray-600 overflow-hidden">
                <div className="px-4 py-3 bg-gray-700 border-b border-gray-600">
                  <h4 className="font-semibold text-white">Results Table</h4>
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-700">
                    <thead className="bg-gray-700">
                      <tr>
                        {Object.keys(results.results[0]).map((header) => (
                          <th
                            key={header}
                            className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider"
                          >
                            {header}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-700">
                      {results.results.map((row, idx) => (
                        <tr key={idx} className="hover:bg-gray-700/50 transition-colors duration-150">
                          {Object.values(row).map((value, valueIdx) => (
                            <td key={valueIdx} className="px-6 py-4 whitespace-nowrap text-gray-300">
                              {value}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalyzeData;