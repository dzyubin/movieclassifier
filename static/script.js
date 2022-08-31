setTimeout(() => {
    startSpinner()
}, 100)

function startSpinner() {
    // face tracking
    const submitBtn = document.querySelector('#submitBtn')

    submitBtn && submitBtn.addEventListener('click', () => {
        const spinner = document.querySelector('div.spinner-grow')
        spinner.style.display = 'inline-block'
        submitBtn.value = 'Tracking faces...'
    })

    // question answering
    const answerQuestionBtn = document.querySelector('#get-answer-btn')
    answerQuestionBtn.addEventListener('click', (evt) => {
        evt.preventDefault()
        const text = document.querySelector('#text').value
        const question = document.querySelector('#question').value

        const spinner = document.querySelector('div.spinner-grow')
        spinner.style.display = 'inline-block'
        answerQuestionBtn.innerHTML = 'Looking for answer...'

        answer.style.animation = 'none'
        score.style.animation = 'none'

        fetch(`/answer?text="${text}"&question="${question}"`)
            .then(res => res.json().then(res1 => {
                answer.innerHTML = res1.answer
                score.innerHTML = res1.score

                answer.style.animation = 'pulse 2s ease alternate'
                score.style.animation = 'pulse 2s ease alternate'
            }))
            .catch(err => console.log(err))
            .then(() => {
                spinner.style.display = 'none'
                answerQuestionBtn.innerHTML = 'Get Answer'
            })
    })
}