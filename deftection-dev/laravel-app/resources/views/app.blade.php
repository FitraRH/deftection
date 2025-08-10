<!DOCTYPE html>
<html data-theme-dark="cinder" data-theme-primary="blue" class="dark">

<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0" />
    <link rel="icon" href="assets/favicon.ico">
    @vite(['resources/css/app.css', 'resources/js/app.js'])
    @routes
    @inertiaHead
</head>

<body data-layout="sideblock">
    <div class="root">
        @inertia
    </div>
</body>

</html>
