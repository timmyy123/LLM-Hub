pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
        maven { url = uri("https://packages.jetbrains.team/maven/p/ki/maven") }
        maven { url = uri("https://packages.jetbrains.team/maven/p/grazi/grazie-platform-public") }
        maven { url = uri("https://raw.githubusercontent.com/NexaAI/core/main") }
    }
}

rootProject.name = "Llm Hub"
include(":app")
 