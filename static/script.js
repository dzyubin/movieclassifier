setTimeout(() => {
    startSpinner()
}, 0)

function startSpinner() {
    const submitBtn = document.querySelector('#submitBtn')

    submitBtn.addEventListener('click', ({ target }) => {
        const spinner = document.querySelector('div.spinner-grow')
        spinner.style.display = 'inline-block'
        submitBtn.value = 'Tracking faces...'
    })
}