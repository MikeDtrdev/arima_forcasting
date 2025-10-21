from api import app

if __name__ == '__main__':
    print("Starting Tondo Crime Forecast API on http://localhost:5000")
    print("Press CTRL+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5000)