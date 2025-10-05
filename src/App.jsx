import "./App.css";
import { useState } from "react";
import { Dropzone } from "@files-ui/react";
import axios from "axios";

const App = () => {
  const [file, setFile] = useState(null);
  const [img, setImg] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (files) => {
    const selectedFile = files[0].file;
    setFile(selectedFile);

    // Create preview URL
    const previewUrl = URL.createObjectURL(selectedFile);
    setPreview(previewUrl);
  };

  const handleUploadToAPI = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await axios.post("http://localhost:8000/ai", formData, {
        responseType: "blob",
      });
      const url = URL.createObjectURL(response.data);
      setImg(url);
    } catch (err) {
      console.error(err);
      alert("Upload failed");
    }
  };

  const handleClear = () => {
    setFile(null);
    setImg(null);
    if (preview) {
      URL.revokeObjectURL(preview);
      setPreview(null);
    }
  };

  return (
    <div className="mainSec">
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "start",
          width: "400px",
          margin: "auto",
        }}
      >
        <h1 style={{ margin: 0, marginTop: "2rem" }}>FiberSense</h1>
        <h3
          style={{
            margin: 0,
            fontSize: "2rem",
            whiteSpace: "nowrap",
            marginBottom: "4rem",
          }}
        >
          Identify textile fibers from a photo
        </h3>
      </div>
      <div className="cardContainer">
        <div className="card">
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              marginBottom: "15px",
              marginTop: "-218px", // Positions it to peek above the card
              marginLeft: "-150px",
            }}
          >
            <img
              src="/cut-it-out.webp" // Replace with your actual image filename
              alt="Earth character"
              style={{
                width: "200px",
                height: "200px",
                objectFit: "contain",
              }}
            />
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              marginBottom: "10px",
            }}
          >
            <img
              alt="upload"
              src="/upload.png"
              style={{
                width: "40px",
                height: "40px",
                marginRight: "10px",
              }}
            />
            <h2 style={{ margin: 0 }}>Upload</h2>
          </div>
          <h4 style={{ margin: 0, marginBottom: "10px" }}>
            Take or upload a photo of the cloth
            <br />
            {"(frontal, well-lit)"}
          </h4>
          <div style={{ height: "167px", marginBottom: "10px" }}>
            {file ? (
              <div
                style={{
                  border: "2px solid #12445e",
                  borderRadius: "10px",
                  overflow: "hidden",
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  backgroundColor: "#f5f5f5",
                }}
              >
                <img
                  src={preview}
                  alt="Preview"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                  }}
                />
              </div>
            ) : (
              <Dropzone
                onChange={handleFileChange}
                accept="image/*"
                maxFileSize={1000 * 1024}
                maxFiles={1}
              />
            )}
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginTop: "2rem",
            }}
          >
            <button
              className="darkButton"
              onClick={handleUploadToAPI}
              disabled={file == null}
            >
              Analyze
            </button>
            <button className="lightButton" onClick={handleClear}>
              Clear
            </button>
          </div>
          <div
            style={{
              width: "fit-content",
              display: "flex",
              marginTop: "1.3rem", // Change this from "5rem" to "1.5rem"
              marginLeft: "auto", // Add this
              marginRight: "auto", // Add this
            }}
          >
            <img
              style={{
                width: "50px",
                height: "50px",
                marginRight: "10px",
              }}
              src="/shirt1.png"
              alt=""
            />
            <img
              style={{
                width: "50px",
                height: "50px",
                marginRight: "10px",
              }}
              src="/shirt2.png"
              alt=""
            />
            <img
              style={{
                width: "50px",
                height: "50px",
                marginRight: "10px",
              }}
              src="/scissors.png"
              alt=""
            />
          </div>
        </div>
        <div className="card">
          <h2 style={{ marginTop: 0 }}>Result</h2>
          {img && (
            <div style={{ width: "100%" }}>
              <img
                style={{ width: "100%", height: "auto" }}
                src={img}
                alt="processed"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
