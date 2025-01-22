<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Checker</title>
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            background-color: #f8f9fa;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        .container {
            max-width: 600px;
            margin-top: 50px;
        }
        .header, .footer {
            background-color: #007bff;
            color: white;
            padding: 10px 0;
            text-align: center;
        }
        .footer {
            margin-top: auto;
        }
    </style>
</head>
<body>

<div class="header">
    <h2>Welcome To News Detector</h2>
</div>

<div class="container">
    <h2 class="text-center">Fake News Checker</h2>
    <form action="index.php" method="post">
        <div class="form-group">
            <label for="newsTitle">Enter News Title:</label>
            <input type="text" class="form-control" id="newsTitle" name="newsTitle" placeholder="Enter a news title" value="<?php echo isset($_POST['newsTitle']) ? htmlspecialchars($_POST['newsTitle']) : ''; ?>" required>
        </div>
        <button type="submit" class="btn btn-primary btn-block">Check</button>
    </form>
    <br>
    <?php
    if ($_SERVER["REQUEST_METHOD"] == "POST") {
        $title = $_POST['newsTitle'];
        
        // Run Python script with the entered title and get the result
        $command = escapeshellcmd("python3 data.py " . escapeshellarg($title));
        $output = shell_exec($command);

        // Display the result from the Python script
        echo "<div class='alert alert-info text-center' role='alert'>";
        echo "<strong>Prediction: </strong>" . htmlspecialchars($output);
        echo "</div>";
    }
    ?>
</div>

<div class="footer">
    <p>Developed By Muhammad Mustofa Muntaseer</p>
</div>

<!-- Bootstrap JS, Popper.js, and jQuery -->
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.2/dist/umd/popper.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
