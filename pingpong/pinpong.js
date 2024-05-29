const canvas = document.getElementById("pingPongCanvas");
const context = canvas.getContext("2d");

let playerScore = 0;
let aiScore = 0;

const paddleWidth = 10;
const paddleHeight = 100;
const ballRadius = 10;

const player = { x: 0, y: canvas.height / 2 - paddleHeight / 2, dy: 0 };
const ai = { x: canvas.width - paddleWidth, y: canvas.height / 2 - paddleHeight / 2, dy: 0 };
const ball = { x: canvas.width / 2, y: canvas.height / 2, dx: 4, dy: 4 };

document.addEventListener("keydown", keyDownHandler);
document.addEventListener("keyup", keyUpHandler);

function keyDownHandler(e) {
    if (e.key === "Up" || e.key === "ArrowUp") {
        player.dy = -8;
    } else if (e.key === "Down" || e.key === "ArrowDown") {
        player.dy = 8;
    }
}

function keyUpHandler(e) {
    if (e.key === "Up" || e.key === "ArrowUp" || e.key === "Down" || e.key === "ArrowDown") {
        player.dy = 0;
    }
}

function update() {
    // Move player paddle
    player.y += player.dy;
    if (player.y < 0) player.y = 0;
    if (player.y + paddleHeight > canvas.height) player.y = canvas.height - paddleHeight;

    // Move AI paddle
    if (ai.y + paddleHeight / 2 < ball.y) {
        ai.dy = 4;
    } else {
        ai.dy = -4;
    }
    ai.y += ai.dy;
    if (ai.y < 0) ai.y = 0;
    if (ai.y + paddleHeight > canvas.height) ai.y = canvas.height - paddleHeight;

    // Move ball
    ball.x += ball.dx;
    ball.y += ball.dy;

    // Ball collision with top and bottom walls
    if (ball.y - ballRadius < 0 || ball.y + ballRadius > canvas.height) {
        ball.dy *= -1;
    }

    // Ball collision with paddles
    if (
        (ball.x - ballRadius < player.x + paddleWidth && ball.y > player.y && ball.y < player.y + paddleHeight) ||
        (ball.x + ballRadius > ai.x && ball.y > ai.y && ball.y < ai.y + paddleHeight)
    ) {
        ball.dx *= -1;
    }

    // Ball goes out of bounds
    if (ball.x - ballRadius < 0) {
        aiScore++;
        resetBall();
    } else if (ball.x + ballRadius > canvas.width) {
        playerScore++;
        resetBall();
    }

    document.getElementById("playerScore").textContent = `Player: ${playerScore}`;
    document.getElementById("aiScore").textContent = `AI: ${aiScore}`;
}

function resetBall() {
    ball.x = canvas.width / 2;
    ball.y = canvas.height / 2;
    ball.dx = -ball.dx;
}

function draw() {
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Draw player paddle
    context.fillStyle = "white";
    context.fillRect(player.x, player.y, paddleWidth, paddleHeight);

    // Draw AI paddle
    context.fillStyle = "white";
    context.fillRect(ai.x, ai.y, paddleWidth, paddleHeight);

    // Draw ball
    context.beginPath();
    context.arc(ball.x, ball.y, ballRadius, 0, Math.PI * 2);
    context.fillStyle = "white";
    context.fill();
    context.closePath();
}

function gameLoop() {
    update();
    draw();
    requestAnimationFrame(gameLoop);
}

gameLoop();
