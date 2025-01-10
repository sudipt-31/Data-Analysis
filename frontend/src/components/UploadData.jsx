import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

const UploadData = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [error, setError] = useState('');
  const [lastUploadedFile, setLastUploadedFile] = useState(null);

  const onDrop = useCallback(acceptedFiles => {
    setFile(acceptedFiles[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    }
  });

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('https://nltosql-hxdwg2arhdhja4dt.canadacentral-01.azurewebsites.net/api/analysis/', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUploadStatus('File uploaded successfully! You can now analyze your data.');
        setLastUploadedFile(file.name);
        setFile(null);
      } else {
        throw new Error(data.error || 'Error uploading file');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-xl shadow-xl border border-gray-700 p-6 space-y-4">
      <h2 className="text-2xl font-bold text-white mb-6">Upload Your Data</h2>

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200 ${
          isDragActive ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-blue-400 hover:bg-gray-700/50'
        }`}
      >
        <input {...getInputProps()} />
        <div className="space-y-2">
          <p className="text-lg text-gray-300">
            {isDragActive ? 'Drop your file here' : 'Drag & drop your file here'}
          </p>
          <p className="text-sm text-gray-400">
            or click to select a file
          </p>
          {file && (
            <p className="text-sm text-blue-400">
              Selected: {file.name}
            </p>
          )}
          {!file && lastUploadedFile && (
            <div className="mt-4">
              <p className="text-sm text-green-400">
                Last uploaded: {lastUploadedFile}
              </p>
              <p className="text-xs text-gray-400 mt-1">
                Drop a new file to replace
              </p>
            </div>
          )}
        </div>
      </div>

      <button
        onClick={handleFileUpload}
        disabled={loading || !file}
        className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white py-3 px-4 rounded-lg font-medium hover:from-blue-600 hover:to-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
      >
        {loading ? 'Uploading...' : 'Upload File'}
      </button>

      {uploadStatus && (
        <p className="text-green-400 text-sm mt-2">{uploadStatus}</p>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 text-red-400">
          {error}
        </div>
      )}
    </div>
  );
};

export default UploadData;
