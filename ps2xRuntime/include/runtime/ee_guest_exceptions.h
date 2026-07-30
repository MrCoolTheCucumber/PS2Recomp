#pragma once

#include <exception>

struct ThreadExitException final : public std::exception
{
    const char *what() const noexcept override
    {
        return "PS2 Thread Exit";
    }
};
