// DynaGate-FFIA Research Website

document.addEventListener("DOMContentLoaded", () => {

    // Add a small fade-in animation when elements
    // enter the viewport.

    const elements = document.querySelectorAll(
        ".info-card, .method-row, .metric-card, " +
        ".noise-card, .ablation-card, .dataset-stat"
    );

    const observer = new IntersectionObserver(
        (entries) => {

            entries.forEach((entry) => {

                if (entry.isIntersecting) {

                    entry.target.classList.add(
                        "visible"
                    );

                    observer.unobserve(
                        entry.target
                    );
                }

            });

        },
        {
            threshold: 0.08
        }
    );

    elements.forEach((element) => {

        element.classList.add(
            "scroll-hidden"
        );

        observer.observe(element);

    });

});
