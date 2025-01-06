import { useState } from 'react'

export default function App() {
  const [file, setFile] = useState(null)
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')

  const handleFileUpload = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Please select a file')
      return
    }

    setLoading(true)
    setError('')
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:8000/api/analysis/', {
        method: 'POST',
        body: formData,
        
      })

      const data = await response.json()
      
      if (response.ok) {
        setUploadStatus('File uploaded successfully! You can now analyze your data.')
        setFile(null)
      } else {
        throw new Error(data.error || 'Error uploading file')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleAnalysis = async (e) => {
    e.preventDefault()
    if (!query.trim()) {
      setError('Please enter a query')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await fetch('http://localhost:8000/api/analysis/', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query }),
      })

      const data = await response.json()
      
      if (response.ok) {
        setResults(data)
      } else {
        throw new Error(data.error || 'Error analyzing data')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSaveResults = async () => {
    if (!results?.results) return
    
    try {
      const response = await fetch('http://localhost:8000/api/save-results/', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ results: results.results }),
      })

      const data = await response.json()
      
      if (response.ok) {
        alert('Results saved successfully!')
      } else {
        throw new Error(data.error || 'Error saving results')
      }
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 p-8">
  <div className="max-w-4xl mx-auto space-y-8">
    <h1 className="text-3xl font-bold text-center text-white">Data Analysis Dashboard</h1>
    
    {/* File Upload Section */}
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
      <h2 className="text-xl font-semibold mb-4 text-white">Upload Data</h2>
      <form onSubmit={handleFileUpload} className="space-y-4">
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={(e) => setFile(e.target.files[0])}
          className="w-full p-2 border border-gray-600 rounded bg-gray-700 text-white file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-gray-600 file:text-white hover:file:bg-gray-500"
        />
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed"
        >
          {loading ? 'Uploading...' : 'Upload'}
        </button>
      </form>
      {uploadStatus && (
        <p className="mt-2 text-green-400">{uploadStatus}</p>
      )}
    </div>

    {/* Query Section */}
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
      <h2 className="text-xl font-semibold mb-4 text-white">Analyze Data</h2>
      <form onSubmit={handleAnalysis} className="space-y-4">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query in natural language..."
          className="w-full p-2 border border-gray-600 rounded bg-gray-700 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent h-32"
        />
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed"
        >
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>
    </div>

    {/* Error Display */}
    {error && (
      <div className="bg-red-900/50 border border-red-700 text-red-400 p-4 rounded">
        {error}
      </div>
    )}

    {/* Results Section */}
    {results && (
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-white">Results</h2>
          <button
            onClick={handleSaveResults}
            className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700"
          >
            Save Results
          </button>
        </div>
        
        {results.explanation && (
          <div className="mb-4 p-4 bg-gray-700/50 rounded border border-gray-600">
            <div dangerouslySetInnerHTML={{ __html: results.explanation }} className="text-gray-200" />
          </div>
        )}
        
        {results.query && (
          <div className="mb-4 p-4 bg-gray-700/50 rounded border border-gray-600">
            <h3 className="font-semibold mb-2 text-white">Generated Query:</h3>
            <code className="block whitespace-pre-wrap text-blue-400">{results.query}</code>
          </div>
        )}
        
        {results.results && (
          <div className="overflow-x-auto rounded border border-gray-600">
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
              <tbody className="bg-gray-800 divide-y divide-gray-700">
                {results.results.map((row, idx) => (
                  <tr key={idx} className="hover:bg-gray-700">
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
        )}
      </div>
    )}
  </div>
</div>
  )
}