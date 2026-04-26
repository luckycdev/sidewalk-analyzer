from sidewalk_analyzer_web.server import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)